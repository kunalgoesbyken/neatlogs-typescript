/**
 * NeatlogsSpanProcessor — attribute normalization and file logging.
 *
 * Port of Python neatlogs/core/span_processor.py
 *
 * Sits between span creation and export, normalizing attributes and handling
 * file logging. The downstream BatchSpanProcessor + OTLPSpanExporter handles
 * actual transport.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { performance } from 'node:perf_hooks';

import { ROOT_CONTEXT, trace as otelTrace, TraceFlags, SpanStatusCode } from '@opentelemetry/api';
import type { Context, HrTime, SpanContext, Tracer } from '@opentelemetry/api';
import type { ReadableSpan } from '@opentelemetry/sdk-trace-base';
import type { Span as SdkSpan, SpanProcessor } from '@opentelemetry/sdk-trace-base';

import { UnifiedAttributeProcessor } from './attribute-processor.js';
import type { SpanDict } from './attribute-processor.js';
import { getScheduledMask, scheduleMask } from './mask.js';
import { getLogger } from './logger.js';
import { applySessionAttributes } from './session.js';
import { applyEndUserAttributes } from './end-user.js';
import { PromptContext, UserPromptContext } from '../prompt/template.js';
import {
  PROMPT_VARIABLES_KEY,
  PROMPT_TEMPLATE_KEY,
  PROMPT_VERSION_KEY,
  USER_PROMPT_TEMPLATE_KEY,
  USER_PROMPT_VARIABLES_KEY,
} from './context.js';
import { popEntry } from './crewai-task-registry.js';
import type { MaskFunction } from '../types.js';
import { AttributeMapper } from '../config/attribute-mapper.js';
import { getNeatlogsTracer } from './provider.js';
import { verificationMarkerFromEnv } from './resource.js';
import { setMediaCaptureAvailability } from './media.js';
import { isHttpSpan } from './http-span.js';
import { captureSpanStarted, captureSpanEnded } from '../dataset/capture-scope.js';
import { stampReplaySpan } from '../replay/session.js';

const logger = getLogger();

function resolveLogFilePath(configuredPath: string): string {
  return path.isAbsolute(configuredPath)
    ? configuredPath
    : path.join(process.cwd(), configuredPath);
}

function createLogStream(configuredPath: string): fs.WriteStream | null {
  const logPath = resolveLogFilePath(configuredPath);
  let fd: number | null = null;

  try {
    fs.mkdirSync(path.dirname(logPath), { recursive: true });
    fd = fs.openSync(logPath, 'a');
    const stream = fs.createWriteStream(logPath, { fd, autoClose: true });
    fd = null;
    stream.on('error', (error) => {
      logger.warn(`Failed to write span log file ${logPath}: ${error}`);
      stream.destroy();
    });
    return stream;
  } catch (error) {
    if (fd !== null) {
      try {
        fs.closeSync(fd);
      } catch {
        // ignore cleanup failure; the open error below is the actionable one.
      }
    }
    logger.warn(`Failed to open span log file ${logPath}: ${error}`);
    return null;
  }
}

const CLOSE_STREAM_TIMEOUT_MS = 5_000;

async function closeLogStream(stream: fs.WriteStream | null, description: string): Promise<void> {
  if (!stream || stream.destroyed || stream.writableEnded) {
    return;
  }

  await new Promise<void>((resolve) => {
    let settled = false;
    const settle = () => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      stream.off('close', closeHandler);
      stream.off('error', errorHandler);
      resolve();
    };

    const closeHandler = settle;
    const errorHandler = (error: Error) => {
      logger.warn(`Failed to close ${description} log file handle: ${error}`);
      settle();
    };
    const timer = setTimeout(() => {
      logger.warn(`Timed out waiting for ${description} log stream to close; destroying`);
      stream.destroy();
      settle();
    }, CLOSE_STREAM_TIMEOUT_MS);

    stream.once('close', closeHandler);
    stream.once('error', errorHandler);

    try {
      stream.end();
    } catch (error) {
      logger.warn(`Failed to close ${description} log file handle: ${error}`);
      settle();
    }
  });
}

// ────────────────────────────────────────────────────────
// LLM span detection patterns
// ────────────────────────────────────────────────────────

const LLM_NAME_PATTERNS = /chat|completion|generate|embedding/i;

// ────────────────────────────────────────────────────────
// Performance stats
// ────────────────────────────────────────────────────────

interface PerfStats {
  onStartTime: number;
  onEndTime: number;
  spansProcessed: number;
  spansExported: number;
}

// ────────────────────────────────────────────────────────
// Helper: convert HrTime ([seconds, nanoseconds]) to ns
// ────────────────────────────────────────────────────────

function hrTimeToNanos(hr: HrTime): number {
  return hr[0] * 1_000_000_000 + hr[1];
}

// ────────────────────────────────────────────────────────
// Helper: convert ReadableSpan to JSON-serializable dict
//
// When includeScope=true, includes instrumentation_scope
// and uses `status.message` (for the UnifiedAttributeProcessor).
// When false (default), uses `status.description` for
// backward-compatible raw logging.
// ────────────────────────────────────────────────────────

export function spanToDict(
  span: ReadableSpan,
  options?: { includeScope?: boolean },
): Record<string, any> {
  const ctx = span.spanContext();
  const includeScope = options?.includeScope ?? false;

  const dict: Record<string, any> = {
    trace_id: ctx.traceId,
    span_id: ctx.spanId,
    parent_span_id: span.parentSpanId ?? null,
    name: span.name,
    kind: span.kind,
    start_time: hrTimeToNanos(span.startTime),
    end_time: hrTimeToNanos(span.endTime),
    attributes: { ...span.attributes },
    resource: span.resource?.attributes ? { ...span.resource.attributes } : {},
    status: includeScope
      ? { code: span.status.code, message: span.status.message ?? undefined }
      : { code: span.status.code, description: span.status.message ?? null },
    events: span.events
      ? span.events.map((e) => ({
          name: e.name,
          timestamp: hrTimeToNanos(e.time),
          attributes: e.attributes ? { ...e.attributes } : {},
        }))
      : [],
  };

  if (includeScope) {
    dict.instrumentation_scope = span.instrumentationLibrary
      ? {
          name: span.instrumentationLibrary.name,
          version: span.instrumentationLibrary.version,
        }
      : null;
  }

  return dict;
}

// ────────────────────────────────────────────────────────
// Constructor options
// ────────────────────────────────────────────────────────

export interface NeatlogsSpanProcessorOptions {
  debug?: boolean;
  mask?: MaskFunction;
  /** Pre-built AttributeMapper. If not supplied, a default mapper is created. */
  mapper?: AttributeMapper;
  emitCompletionMarkers?: boolean;
  /** Treat every span on this processor's private provider as SDK-owned. */
  ownAllSpans?: boolean;
  /** Gate raw typed-media staging for this processor's export pipeline. */
  mediaUploadsAvailable?: boolean;
  mediaUploadsUnavailableReason?: string;
  mediaOwner?: object;
}

// ────────────────────────────────────────────────────────
// NeatlogsSpanProcessor
// ────────────────────────────────────────────────────────

export class NeatlogsSpanProcessor implements SpanProcessor {
  private readonly debug: boolean;
  private readonly mask: MaskFunction | undefined;
  private readonly unifiedProcessor: UnifiedAttributeProcessor;
  private readonly emitCompletionMarkers: boolean;
  private readonly ownAllSpans: boolean;
  private readonly mediaUploadsAvailable: boolean;
  private readonly mediaUploadsUnavailableReason: string;
  private readonly mediaOwner: object | undefined;

  private perfStats: PerfStats;
  private _retrieversToSuppress: Set<string>;
  private _activeSpans: Map<string, SdkSpan>;
  private _closingReason: string | null;
  private _closed: boolean;
  private _completionEligibleRoots: WeakSet<object>;
  private _pendingMasks: Set<Promise<unknown>>;

  // File logging
  private _logRawSpansEnabled: boolean;
  private _logProcessedSpansEnabled: boolean;
  private _rawLogStream: fs.WriteStream | null;
  private _processedLogStream: fs.WriteStream | null;

  constructor(opts: NeatlogsSpanProcessorOptions = {}) {
    this.debug = opts.debug ?? false;
    this.mask = opts.mask;
    this.emitCompletionMarkers = opts.emitCompletionMarkers ?? true;
    this.ownAllSpans = opts.ownAllSpans ?? false;
    this.mediaUploadsAvailable = opts.mediaUploadsAvailable ?? false;
    this.mediaUploadsUnavailableReason =
      opts.mediaUploadsUnavailableReason ?? 'telemetry_uploads_disabled';
    this.mediaOwner = opts.mediaOwner;

    this.unifiedProcessor = new UnifiedAttributeProcessor(
      opts.mapper ?? new AttributeMapper(),
      this.debug,
    );

    this._logRawSpansEnabled = false;
    this._logProcessedSpansEnabled = false;
    this._rawLogStream = null;
    this._processedLogStream = null;

    this._initFileLogging();

    this.perfStats = {
      onStartTime: 0,
      onEndTime: 0,
      spansProcessed: 0,
      spansExported: 0,
    };

    this._retrieversToSuppress = new Set<string>();
    this._activeSpans = new Map<string, SdkSpan>();
    this._closingReason = null;
    this._closed = false;
    this._completionEligibleRoots = new WeakSet<object>();
    this._pendingMasks = new Set<Promise<unknown>>();
  }

  // ── File logging init ─────────────────────────────────

  private _initFileLogging(): void {
    // Raw OTel span JSON (before attribute normalization)
    this._logRawSpansEnabled =
      this.debug ||
      ['true', '1', 'yes'].includes((process.env.NEATLOGS_LOG_RAW_SPANS ?? '').toLowerCase());

    if (this._logRawSpansEnabled && this.mask) {
      logger.warn(
        'Raw span logging is disabled while a mask is configured because raw files bypass redaction.',
      );
      this._logRawSpansEnabled = false;
    }

    if (this._logRawSpansEnabled) {
      const rawPath = process.env.NEATLOGS_LOG_RAW_SPANS_FILE ?? 'spans_raw_optimized.log';
      this._rawLogStream = createLogStream(rawPath);
      if (this._rawLogStream) {
        logger.info(`Raw span logging enabled: ${resolveLogFilePath(rawPath)}`);
      }
    }

    // Processed span dict (after normalization)
    this._logProcessedSpansEnabled = ['true', '1', 'yes'].includes(
      (process.env.NEATLOGS_LOG_SPANS ?? '').toLowerCase(),
    );

    if (this._logProcessedSpansEnabled) {
      const processedPath = process.env.NEATLOGS_LOG_SPANS_FILE ?? 'spans_optimized.log';
      this._processedLogStream = createLogStream(processedPath);
      if (this._processedLogStream) {
        logger.info(`Processed span logging enabled: ${resolveLogFilePath(processedPath)}`);
      }
    }
  }

  // ── SpanProcessor.onStart ─────────────────────────────

  onStart(span: SdkSpan, parentContext: Context): void {
    if (this._closed) return;
    const startTime = performance.now();
    try {
      const scopeName = span.instrumentationLibrary?.name ?? '';
      const isCompletionMarker = span.name === 'neatlogs.trace.complete';
      // Auto-instrumented framework children retain their own scope name. An
      // active Neatlogs parent makes them part of the SDK-owned trace, while a
      // foreign root on a caller-supplied provider remains untouched.
      const sdkOwned =
        this.ownAllSpans ||
        scopeName.startsWith('neatlogs') ||
        (span.parentSpanId !== undefined && this._activeSpans.has(span.parentSpanId));
      if (sdkOwned && !isCompletionMarker) {
        stampReplaySpan(span);
        captureSpanStarted(span, this);
        setMediaCaptureAvailability(
          span as object,
          this.mediaUploadsAvailable,
          this.mediaUploadsUnavailableReason,
          this.mediaOwner,
        );
        this._activeSpans.set(span.spanContext().spanId, span);
        const verificationMarker = verificationMarkerFromEnv();
        if (verificationMarker) {
          span.setAttribute('neatlogs.verification.marker', verificationMarker);
        }
      }
      if (sdkOwned && !isCompletionMarker && this._closingReason) {
        this._markInterrupted(span, this._closingReason);
        span.end();
        return;
      }

      // Stamp request-scoped identity (identify()) onto ANY root span as a
      // fallback. trace()/span() set it explicitly (overriding this); direct
      // wrappers' auto-roots stamp it themselves. This catch-all is what lets
      // identify() reach FRAMEWORK roots (openai-agents/strands/pi-agent) that
      // open their own root without stamping identity. No-ops when identify() is
      // inactive; child spans are skipped.
      if (!span.parentSpanId) {
        applySessionAttributes(span, undefined, true);
        applyEndUserAttributes(span, undefined, undefined, true);
      }

      const attrs = (span as any).attributes ?? {};
      const spanKind = attrs['openinference.span.kind'] as string | undefined;
      const spanName: string = typeof (span as any).name === 'string' ? (span as any).name : '';

      const isLlmSpan = spanKind === 'LLM' || LLM_NAME_PATTERNS.test(spanName);

      if (!isLlmSpan) return;

      // Read context values only from the span's OWN parentContext. The global
      // OTel context belongs to another SDK and must never be consulted.
      const getFrom = (ctx: Context | undefined, key: symbol): string | undefined =>
        typeof ctx?.getValue === 'function' ? (ctx.getValue(key) as string | undefined) : undefined;
      const readValue = (key: symbol): string | undefined => getFrom(parentContext, key);
      let variablesJson = readValue(PROMPT_VARIABLES_KEY);
      let template = readValue(PROMPT_TEMPLATE_KEY);
      const versionVal = readValue(PROMPT_VERSION_KEY);
      let userTemplate = readValue(USER_PROMPT_TEMPLATE_KEY);
      let userVariablesJson = readValue(USER_PROMPT_VARIABLES_KEY);

      // Fall back to PromptContext / UserPromptContext
      if (!variablesJson) {
        const captured = PromptContext.getVariables();
        if (captured) {
          variablesJson = JSON.stringify(captured);
        }
      }

      if (!template) {
        const captured = PromptContext.getTemplate();
        if (captured) {
          template = captured;
        }
      }

      if (!userTemplate) {
        const captured = UserPromptContext.getTemplate();
        if (captured) {
          userTemplate = captured;
        }
      }

      if (!userVariablesJson) {
        const captured = UserPromptContext.getVariables();
        if (captured) {
          userVariablesJson = JSON.stringify(captured);
        }
      }

      if (this.debug) {
        logger.debug(`[SpanProcessor.onStart] LLM span '${spanName}' starting`);
        logger.debug(`  variables_json from context: ${variablesJson}`);
        logger.debug(`  template from context: ${template}`);
        logger.debug(`  version from context: ${versionVal}`);
        logger.debug(`  user_template from context: ${userTemplate}`);
        logger.debug(`  user_variables_json from context: ${userVariablesJson}`);
      }

      if (variablesJson) {
        span.setAttribute('llm.prompt_template_variables', variablesJson);
      }
      if (template) {
        span.setAttribute('llm.prompt_template', template);
      }
      if (versionVal) {
        span.setAttribute('llm.prompt_template.version', versionVal);
      }
      if (userTemplate) {
        span.setAttribute('llm.user_prompt_template', userTemplate);
      }
      if (userVariablesJson) {
        span.setAttribute('llm.user_prompt_template_variables', userVariablesJson);
      }
    } finally {
      this.perfStats.onStartTime += performance.now() - startTime;
    }
  }

  // ── SpanProcessor.onEnd ───────────────────────────────

  onEnd(span: ReadableSpan): void {
    captureSpanEnded(span);
    this._activeSpans.delete(span.spanContext().spanId);
    if (this._closed) return;

    // Transport spans are never part of NeatLogs telemetry. Suppress them
    // before logging, normalization, root I/O backfill, masking, or markers.
    if (isHttpSpan(span)) return;

    // Completion markers skip normalization but still cross the same global
    // masking boundary: they carry root identity and metadata.
    if (span.name === 'neatlogs.trace.complete') {
      const markerData = spanToDict(span, { includeScope: true });
      markerData.resource = { attributes: markerData.resource };
      const maskResult = scheduleMask(span as object, markerData, this.mask ?? null);
      const pending = maskResult.finally(() => {
        this._pendingMasks.delete(pending);
      });
      this._pendingMasks.add(pending);
      return;
    }

    const startTime = performance.now();
    this.perfStats.spansProcessed += 1;

    try {
      if (this.debug) {
        logger.debug(`[SpanProcessor.onEnd] Span ending: ${span.name}`);
      }

      // 1. Log raw OTel span (before any processing)
      if (this._rawLogStream && !this._rawLogStream.destroyed) {
        try {
          this._rawLogStream.write(JSON.stringify(spanToDict(span)) + '\n');
        } catch (e) {
          logger.warn(`Failed to write span to raw log file: ${e}`);
        }
      }

      // 2. Process and normalize attributes
      const spanDict = spanToDict(span, { includeScope: true }) as SpanDict;
      const unifiedAttrs = this.unifiedProcessor.normalize(spanDict);

      // 3. Filter large tokenized arrays for EMBEDDING/VECTOR_STORE spans
      let nlKind = unifiedAttrs['neatlogs.span.kind'] as string | undefined;
      if (nlKind === 'embedding' || nlKind === 'vector_store') {
        const skipOutput = unifiedAttrs['neatlogs._skip_output_value'] === true;
        const keysToRemove: string[] = [];

        for (const key of Object.keys(unifiedAttrs)) {
          if (
            key.includes('input_messages') ||
            key.includes('output_messages') ||
            key.includes('gen_ai.prompt') ||
            key.includes('gen_ai.completion') ||
            key.includes('.content')
          ) {
            keysToRemove.push(key);
          } else if (
            key === 'neatlogs.embedding.output' ||
            (skipOutput && key === 'neatlogs.embedding.input')
          ) {
            keysToRemove.push(key);
          }
        }

        for (const key of keysToRemove) {
          delete unifiedAttrs[key];
        }

        if (this.debug && keysToRemove.length > 0) {
          logger.debug(
            `[EMBEDDING Filter] Removed ${keysToRemove.length} large attribute keys ` +
              `from ${nlKind} span (skip_output=${skipOutput})`,
          );
        }
      }

      // 4. Filter prompt template keys for non-LLM spans
      nlKind = unifiedAttrs['neatlogs.span.kind'] as string | undefined;
      if (
        nlKind !== 'llm' &&
        nlKind !== 'embedding' &&
        nlKind !== 'crewai_task' &&
        span.name !== 'PromptTemplate'
      ) {
        delete unifiedAttrs['neatlogs.llm.prompt_template'];
        delete unifiedAttrs['neatlogs.llm.prompt_template_variables'];
        delete unifiedAttrs['neatlogs.llm.prompt_template.version'];
      }

      // PromptTemplate spans → internal
      if (span.name === 'PromptTemplate') {
        if (unifiedAttrs['neatlogs.internal'] === undefined) {
          unifiedAttrs['neatlogs.internal'] = true;
        }
        unifiedAttrs['neatlogs.span.kind'] = 'Neatlogs.INTERNAL';
      }

      // 4b. Retriever span dedup
      if (nlKind === 'retriever') {
        const isInternal = unifiedAttrs['neatlogs.internal'] === true;
        if (isInternal && span.parentSpanId) {
          this._retrieversToSuppress.add(span.parentSpanId);
        }
        const spanId = span.spanContext().spanId;
        if (this._retrieversToSuppress.has(spanId)) {
          this._retrieversToSuppress.delete(spanId);
          unifiedAttrs['neatlogs.internal'] = true;
          if (this.debug) {
            logger.debug(
              `[Retriever Merge] Marked OI retriever '${span.name}' as internal ` +
                `(had neatlogs retriever child)`,
            );
          }
        }
      }

      // 5. Build resource attributes
      const resourceAttrs: Record<string, any> = {};
      if (span.resource?.attributes) {
        for (const [key, value] of Object.entries(span.resource.attributes)) {
          if (
            typeof value === 'string' ||
            typeof value === 'number' ||
            typeof value === 'boolean'
          ) {
            resourceAttrs[key] = value;
          } else if (Array.isArray(value)) {
            resourceAttrs[key] = [...value];
          } else {
            resourceAttrs[key] = String(value);
          }
        }

        if (this.debug && 'neatlogs.tags' in resourceAttrs) {
          logger.debug(
            `[Tags] Span ${span.name}: resource.neatlogs.tags = ${resourceAttrs['neatlogs.tags']}`,
          );
        }
      }

      // Format IDs — JS OTel SDK already provides hex strings
      const traceId = span.spanContext().traceId;
      const spanId = span.spanContext().spanId;
      let parentSpanId: string | null = span.parentSpanId ?? null;

      // Detect self-parenting
      if (parentSpanId === spanId) {
        if (this.debug) {
          logger.warn(
            `[SpanProcessor] Detected self-parenting span. ` +
              `trace_id=${traceId} span_id=${spanId} name=${span.name}. ` +
              `Setting parent_span_id=None.`,
          );
        }
        parentSpanId = null;
      }

      // 6. Build span_data dict
      const kindValue = (unifiedAttrs['neatlogs.span.kind'] as string) ?? 'UNKNOWN';
      const startTimeNs = hrTimeToNanos(span.startTime);
      const endTimeNs = hrTimeToNanos(span.endTime);

      let spanData: Record<string, any> = {
        trace_id: traceId,
        span_id: spanId,
        parent_span_id: parentSpanId,
        name: span.name,
        kind: kindValue || 'UNKNOWN',
        start_time: startTimeNs,
        end_time: endTimeNs,
        duration_ns: endTimeNs > 0 ? endTimeNs - startTimeNs : null,
        attributes: unifiedAttrs,
        resource: { attributes: resourceAttrs },
        status: {
          code: SpanStatusCode[span.status.code] ?? String(span.status.code),
          description: span.status.message ?? null,
        },
        links: span.links.map((link) => ({
          trace_id: link.context.traceId,
          span_id: link.context.spanId,
          trace_flags: link.context.traceFlags,
          trace_state: link.context.traceState?.serialize(),
          attributes: link.attributes ? { ...link.attributes } : {},
        })),
        events: span.events
          ? span.events.map((e) => ({
              name: e.name,
              timestamp: hrTimeToNanos(e.time),
              attributes: e.attributes ? { ...e.attributes } : {},
            }))
          : [],
      };

      // 7. Post-processing: framework span name normalization, CrewAI tasks, model resolution
      let results = this._normalizeFrameworkSpanNames([spanData]);
      spanData = results[0] ?? spanData;
      results = this._injectCrewaiTaskTemplates([spanData]);
      spanData = results[0] ?? spanData;
      this._resolveActualModelName(spanData);

      // 7b. Write normalized neatlogs.* attributes back before the downstream
      // BatchSpanProcessor snapshots the span. Masking is applied later to an
      // exporter-owned clone, so async user callbacks never block onEnd().
      const finalAttrs = (spanData.attributes ?? {}) as Record<string, any>;
      try {
        const spanAttrs = (span as any).attributes ?? (span as any)._attributes;
        if (spanAttrs != null) {
          for (const [k, v] of Object.entries(finalAttrs)) {
            if (typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') {
              spanAttrs[k] = v;
            } else if (
              Array.isArray(v) &&
              v.every(
                (i: any) =>
                  typeof i === 'string' || typeof i === 'number' || typeof i === 'boolean',
              )
            ) {
              spanAttrs[k] = [...v];
            }
          }
        }
      } catch (wbExc) {
        if (this.debug) {
          logger.debug(`[SpanProcessor] Attr write-back failed: ${wbExc}`);
        }
      }

      // 7c. Mask asynchronously. FilteringExporter awaits this exact promise
      // and exports a clone built only from the accepted result. Exceptions are
      // fail-closed in scheduleMask/applyMask.
      const maskResult = scheduleMask(span as object, spanData, this.mask ?? null);
      const pending = maskResult
        .then((maskedSpanData) => {
          if (maskedSpanData === null) return;
          if (isHttpSpan(span, maskedSpanData.attributes ?? {})) return;

          // Processed diagnostic logs follow the same masking boundary as OTLP.
          if (this._processedLogStream && !this._processedLogStream.destroyed) {
            try {
              this._processedLogStream.write(JSON.stringify(maskedSpanData) + '\n');
            } catch (e) {
              logger.warn(`Failed to write span to processed log file: ${e}`);
            }
          }

          this.perfStats.spansExported += 1;
          if (!span.parentSpanId) {
            this._completionEligibleRoots.add(span as object);
            if (this.emitCompletionMarkers) {
              this.emitCompletionMarker(span, traceId, resourceAttrs);
            }
          }
        })
        .finally(() => {
          this._pendingMasks.delete(pending);
        });
      this._pendingMasks.add(pending);
    } finally {
      this.perfStats.onEndTime += performance.now() - startTime;
    }
  }

  /**
   * End active Neatlogs-owned spans child-first before provider shutdown.
   * Ending the actual root through the normal lifecycle emits the completion
   * marker; interruption attributes keep the termination explicit.
   */
  endActiveSpans(reason = 'shutdown'): number {
    // Claim the current registry synchronously before ending anything. Span
    // end calls onEnd synchronously, and shutdown can be entered again from a
    // host callback; swapping the map first makes those paths idempotent.
    const cleanReason = this.beginShutdown(reason);
    const activeById = this._activeSpans;
    this._activeSpans = new Map<string, SdkSpan>();
    const active = [...activeById.values()];
    if (active.length === 0) return 0;

    const activeIds = new Set(active.map((span) => span.spanContext().spanId));
    const depth = (span: SdkSpan): number => {
      let current: SdkSpan | undefined = span;
      const seen = new Set<string>();
      let value = 0;
      while (current?.parentSpanId && activeIds.has(current.parentSpanId)) {
        const parentId = current.parentSpanId;
        if (seen.has(parentId)) break;
        seen.add(parentId);
        value += 1;
        current = activeById.get(parentId);
      }
      return value;
    };

    let ended = 0;
    for (const span of active.sort((a, b) => depth(b) - depth(a))) {
      try {
        if (!span.isRecording()) continue;
        this._markInterrupted(span, cleanReason);
        span.end();
        ended += 1;
      } catch (error) {
        logger.warn(`Failed to end active span during ${cleanReason}: ${error}`);
      }
    }
    return ended;
  }

  beginShutdown(reason = 'shutdown'): string {
    const cleanReason = sanitizeTerminationReason(reason);
    this._closingReason ??= cleanReason;
    return this._closingReason;
  }

  ownsSpan(span: ReadableSpan): boolean {
    return this.ownAllSpans || (span.instrumentationLibrary?.name ?? '').startsWith('neatlogs');
  }

  isCompletionEligible(span: ReadableSpan): boolean {
    return !this._closed && this._completionEligibleRoots.has(span as object);
  }

  async whenMaskComplete(span: ReadableSpan): Promise<boolean> {
    const result = getScheduledMask(span as object);
    if (!result) return false;
    return (await result) !== null && !this._closed;
  }

  private _markInterrupted(span: SdkSpan, cleanReason: string): void {
    span.setAttribute('neatlogs.trace.interrupted', true);
    span.setAttribute('neatlogs.trace.termination.reason', cleanReason);
    // Preserve an explicit OK set by application/framework code. An interrupted
    // span that never reached a terminal status is an OTel ERROR; a completed
    // span remains OK and carries the interruption metadata/event separately.
    if (span.status.code === SpanStatusCode.UNSET) {
      span.setStatus({
        code: SpanStatusCode.ERROR,
        message: `Interrupted during ${cleanReason}`,
      });
    }
    span.addEvent('neatlogs.trace.interrupted', {
      'neatlogs.trace.termination.reason': cleanReason,
    });
  }

  // ── Completion marker ─────────────────────────────────

  emitCompletionMarker(
    rootSpan: ReadableSpan,
    traceId: string,
    resourceAttrs: Record<string, any>,
    markerTracer: Tracer = getNeatlogsTracer('neatlogs.internal'),
  ): void {
    try {
      const spanCtx: SpanContext = {
        traceId: rootSpan.spanContext().traceId,
        spanId: rootSpan.spanContext().spanId,
        isRemote: false,
        traceFlags: TraceFlags.SAMPLED,
      };

      // Use wrapSpanContext to create a non-recording span from context
      const wrappedSpan = otelTrace.wrapSpanContext(spanCtx);
      const ctx = otelTrace.setSpan(ROOT_CONTEXT, wrappedSpan);

      const marker = markerTracer.startSpan('neatlogs.trace.complete', undefined, ctx);
      marker.setAttribute('neatlogs.trace.complete', true);
      marker.setAttribute('neatlogs.internal', true);
      marker.setAttribute('neatlogs.span.kind', 'Neatlogs.INTERNAL');

      // The completion marker may be exported in a batch without the root span.
      // Carry the root-owned identity so ingestion can finalize the trace under
      // the correct conversation and end-user without depending on batch order.
      for (const key of [
        'neatlogs.session.id',
        'neatlogs.session.parent_id',
        'neatlogs.session.feature.name',
        'neatlogs.session.entry_point',
        'neatlogs.end_user.id',
        'neatlogs.end_user.metadata',
      ]) {
        const value = rootSpan.attributes[key];
        if (value !== undefined) {
          marker.setAttribute(key, value);
        }
      }

      // Copy resource tags
      if (resourceAttrs['neatlogs.tags']) {
        marker.setAttribute('neatlogs.tags', resourceAttrs['neatlogs.tags']);
      }

      marker.end();

      if (this.debug) {
        logger.debug(`[SpanProcessor] Emitted completion marker for trace ${traceId}`);
      }
    } catch (e) {
      logger.warn(`[SpanProcessor] Failed to emit completion marker: ${e}`);
    }
  }

  // ── Framework span name normalization ─────────────────

  private _normalizeFrameworkSpanNames(spans: Record<string, any>[]): Record<string, any>[] {
    for (const s of spans) {
      const name: string = s.name ?? '';
      const kind: string = s.kind ?? s.attributes?.['neatlogs.span.kind'] ?? '';

      if (kind !== 'task' || !name.endsWith('.task')) {
        continue;
      }

      const attrs: Record<string, any> = s.attributes ?? {};
      if (!Object.keys(attrs).some((k) => k.startsWith('neatlogs.crewai.'))) {
        continue;
      }

      let desc = name.slice(0, -'.task'.length).trimEnd();
      while (desc.endsWith('.')) {
        desc = desc.slice(0, -1).trimEnd();
      }

      if (desc) {
        if (attrs['neatlogs.task.description'] === undefined) {
          attrs['neatlogs.task.description'] = desc;
        }
      }

      s.name = 'crewai.task';
      s.attributes = attrs;
    }

    return spans;
  }

  // ── CrewAI task template injection ────────────────────

  private _injectCrewaiTaskTemplates(spans: Record<string, any>[]): Record<string, any>[] {
    for (const s of spans) {
      const attrs: Record<string, any> = s.attributes ?? {};
      const taskId = attrs['neatlogs.task.id'];
      if (!taskId) continue;

      const entry = popEntry(String(taskId));
      if (!entry) continue;

      const [tplStr, varsJson] = entry;
      attrs['neatlogs.task.user_prompt_template'] = tplStr;
      if (varsJson) {
        attrs['neatlogs.task.user_prompt_template_variables'] = varsJson;
      }
      attrs['neatlogs.span.kind'] = 'crewai_task';
      s.attributes = attrs;
    }

    return spans;
  }

  // ── Resolve actual model name from LLM output ────────
  // LangChain/Azure spans report deployment names (e.g., "gpt-3.5-turbo") as
  // the model name. The actual resolved model (e.g., "gpt-5-nano-2025-08-07")
  // is buried in the LLM output JSON at response_metadata.model_name.

  private _resolveActualModelName(spanData: Record<string, any>): void {
    const attrs: Record<string, any> = spanData.attributes ?? {};
    const kind = attrs['neatlogs.span.kind'];
    if (kind !== 'llm') return;

    const currentModel = attrs['neatlogs.llm.model_name'] as string | undefined;
    if (!currentModel) return;

    const llmOutput = attrs['neatlogs.llm.output'] as string | undefined;
    if (!llmOutput) return;

    // Plenty of integrations put plain assistant text here — there is no model
    // name to recover from that, so don't try to parse it as JSON.
    const trimmed = llmOutput.trimStart();
    if (!trimmed.startsWith('{') && !trimmed.startsWith('[')) return;

    try {
      const output = JSON.parse(llmOutput);

      // LangChain format: generations[0][0].message.kwargs.response_metadata.model_name
      const generations = output?.generations;
      if (Array.isArray(generations) && generations[0]?.[0]) {
        const respMeta = generations[0][0]?.message?.kwargs?.response_metadata;
        if (respMeta?.model_name && respMeta.model_name !== currentModel) {
          attrs['neatlogs.llm.model_name'] = respMeta.model_name;
          if (this.debug) {
            logger.debug(`[ModelResolve] LangChain span: ${currentModel} → ${respMeta.model_name}`);
          }
          return;
        }
      }

      // Vercel AI SDK / direct OpenAI format: model field at top level of response
      if (output?.model && output.model !== currentModel) {
        attrs['neatlogs.llm.model_name'] = output.model;
        if (this.debug) {
          logger.debug(`[ModelResolve] Direct response: ${currentModel} → ${output.model}`);
        }
      }
    } catch (e: unknown) {
      // Best-effort enrichment only — never worth warning the user about.
      if (this.debug) {
        logger.debug(`[ModelResolve] Failed to parse LLM output for model extraction: ${e}`);
      }
    }
  }

  // ── forceFlush / shutdown ─────────────────────────────

  async forceFlush(): Promise<void> {
    // Mask completion may create a trace-completion marker, so drain until no
    // newly-created masking work remains before transport processors flush.
    while (this._pendingMasks.size > 0) {
      await Promise.allSettled([...this._pendingMasks]);
    }
  }

  async shutdown(): Promise<void> {
    await this.forceFlush();
    this._closed = true;
    this._logPerformanceStats();

    await closeLogStream(this._rawLogStream, 'raw');
    this._rawLogStream = null;

    await closeLogStream(this._processedLogStream, 'processed');
    this._processedLogStream = null;
  }

  // ── Performance stats ─────────────────────────────────

  private _logPerformanceStats(): void {
    if (!this.debug) return;

    const stats = this.perfStats;
    if (stats.spansProcessed === 0) return;

    const totalTime = stats.onStartTime + stats.onEndTime;
    const avgMs = totalTime / stats.spansProcessed;

    try {
      logger.info(
        `Neatlogs overhead: ${totalTime.toFixed(2)}ms total, ${avgMs.toFixed(3)}ms/span ` +
          `(${stats.spansProcessed} spans processed, ` +
          `${stats.spansExported} spans logged)`,
      );
    } catch {
      // ignore
    }
  }

  // ── Accessors for testing ─────────────────────────────

  /** @internal — exposed for testing */
  get _perfStats(): PerfStats {
    return this.perfStats;
  }

  /** @internal — exposed for testing */
  get _suppressedRetrievers(): Set<string> {
    return this._retrieversToSuppress;
  }
}

export function sanitizeTerminationReason(reason: unknown): string {
  const printable = String(reason || 'shutdown')
    .replace(/[\u0000-\u001f\u007f-\u009f]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  return (printable || 'shutdown').slice(0, 256);
}

export class CompletionMarkerSpanProcessor implements SpanProcessor {
  private closed = false;
  private deferring = false;
  private readonly deferredRoots: ReadableSpan[] = [];
  private readonly pending = new Set<Promise<void>>();

  constructor(
    private readonly source: NeatlogsSpanProcessor,
    private readonly tracer: Tracer,
  ) {}

  onStart(_span: SdkSpan, _parentContext: Context): void {}

  onEnd(span: ReadableSpan): void {
    if (
      this.closed ||
      span.name === 'neatlogs.trace.complete' ||
      span.parentSpanId ||
      isHttpSpan(span)
    )
      return;
    if (!this.source.ownsSpan(span)) return;
    const task = this.source
      .whenMaskComplete(span)
      .then((eligible) => {
        if (!eligible || this.closed) return;
        if (this.deferring) this.deferredRoots.push(span);
        else this.emit(span);
      })
      .finally(() => {
        this.pending.delete(task);
      });
    this.pending.add(task);
  }

  beginShutdown(): void {
    this.deferring = true;
  }

  emitDeferred(): void {
    const roots = this.deferredRoots.splice(0);
    // The log-drain boundary has passed. Roots ending after this point must
    // emit immediately rather than being stranded in a second deferred batch.
    this.deferring = false;
    for (const span of roots) this.emit(span);
  }

  private emit(span: ReadableSpan): void {
    const resourceAttrs = { ...span.resource.attributes };
    this.source.emitCompletionMarker(span, span.spanContext().traceId, resourceAttrs, this.tracer);
  }

  async forceFlush(): Promise<void> {
    while (this.pending.size > 0) {
      await Promise.allSettled([...this.pending]);
    }
    this.emitDeferred();
  }
  async shutdown(): Promise<void> {
    await this.forceFlush();
    this.closed = true;
    this.deferredRoots.splice(0);
  }
}
