/**
 * Neatlogs SDK initialisation, flush, and shutdown.
 *
 * Port of Python neatlogs/init.py to TypeScript.
 *
 * `init()` sets up the OTel TracerProvider, MeterProvider, LoggerProvider,
 * span processors, and exporters.
 * `flush()` and `shutdown()` handle graceful cleanup.
 */

import * as path from "node:path";
import { createHash } from "node:crypto";
import { configureReplayFromEnvironment } from './replay/environment.js';

import { Resource } from "@opentelemetry/resources";
import { ATTR_SERVICE_NAME } from "@opentelemetry/semantic-conventions";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import {
  ParentBasedSampler,
  TraceIdRatioBasedSampler,
  type BasicTracerProvider,
  type SpanProcessor,
  type SpanExporter,
} from "@opentelemetry/sdk-trace-base";
import { ExportResultCode } from "@opentelemetry/core";
import { OTLPTraceExporter } from "./core/otlp-trace-exporter.js";
import { CompressionAlgorithm } from "@opentelemetry/otlp-exporter-base";
import { LoggerProvider } from "@opentelemetry/sdk-logs";
import { OTLPLogExporter } from "@opentelemetry/exporter-logs-otlp-proto";

import {
  CompletionMarkerSpanProcessor,
  NeatlogsSpanProcessor,
} from "./core/span-processor.js";
import { addVerificationMarkerResourceAttribute } from "./core/resource.js";
import { spanLimitsForCaptureEverything } from "./constants.js";
import { getRegisteredClients } from "./core/client-registry.js";
import { registerCaptureProvider } from './dataset/capture-scope.js';
import { FilteringExporter } from "./core/filtering-exporter.js";
import { capturePreparedSpans, clearDoctorCapture } from "./core/doctor-capture.js";
import { ByteLimitedSpanExporter } from "./core/byte-limited-exporter.js";
import { MaskingLogExporter } from "./core/masking-log-exporter.js";
import { discardPendingMediaOwner } from "./core/media.js";
import {
  DeliveryDiagnostics,
  type DeliveryDiagnosticsSnapshot,
} from "./core/delivery-diagnostics.js";
import {
  ObservableBatchLogRecordProcessor,
  ObservableBatchSpanProcessor,
} from "./core/observable-batch-processors.js";
import { _setOtelLogger } from "./core/log.js";
import { _setSessionConfig } from "./core/context.js";
import { _setNeatlogsProvider } from "./core/provider.js";
import { getLogger, enableDebugLogging } from "./core/logger.js";
import {
  PromptClient,
  closeSharedClient,
  setSharedClient,
} from "./prompt/client.js";
import { _resetMastraCache } from "./mastra.js";
import { __version__ } from "./version.js";
import { NeatlogsConfigurationError } from "./errors.js";
import { DEFAULT_INGEST_ENDPOINT, exportQueueCapacity } from "./constants.js";
import { runByDeadline } from "./core/deadline.js";
import {
  DisabledUploadAuthority,
  isUploadAuthority,
  resolveUploadAuthority,
  uploadsEnabledFromEnv,
  type UploadAuthorityOption,
} from "./core/upload-authority.js";
import type { InitOptions } from "./types.js";
import { AbTestClient } from './ab-test/client.js';
import { getAbTestClient, setAbTestClient } from './ab-test/shared.js';

const logger = getLogger();

// ---------------------------------------------------------------------------
// Module-level state
// ---------------------------------------------------------------------------

let _initialized = false;
type LifecycleState = "uninitialized" | "initializing" | "running" | "closing";
let _lifecycleState: LifecycleState = "uninitialized";
let _initPromise: Promise<void> | null = null;
let _tracerProvider: BasicTracerProvider | null = null;
let _ownsTracerProvider = false;
let _logProvider: LoggerProvider | null = null;
let _spanProcessor: NeatlogsSpanProcessor | null = null;
let _transportSpanProcessors: SpanProcessor[] = [];
let _completionProcessor: CompletionMarkerSpanProcessor | null = null;
let _debugMode = false;
let _deliveryDiagnostics = new DeliveryDiagnostics();
interface InitIdentity {
  serialized: string;
  mask: InitOptions["mask"];
  tracerProvider: InitOptions["tracerProvider"];
  uploadAuthority: UploadAuthorityOption | undefined;
  doctorProbeExporter: InitOptions["doctorProbeExporter"];
}
let _initIdentity: InitIdentity | null = null;
let _effectiveSampleRate = 1;
let _exportEnabled = false;
let _queueMaxSize = 2_048;

// Track signal handlers so we can remove them in shutdown()
let _sigHandlersRegistered = false;
let _signalShutdownStarted = false;
let _shutdownPromise: Promise<boolean> | null = null;
const _shutdownBeforeExit = () => {
  void shutdown("beforeExit").catch((error) => {
    logger.error(`Error shutting down before exit: ${error}`);
  });
};
const _shutdownOnSignal = (signal: NodeJS.Signals) => {
  if (_signalShutdownStarted) return;
  _signalShutdownStarted = true;
  // Node delivered this original signal to every registered listener. If the
  // host has one, it owns termination; re-sending would look like a second
  // signal. Only restore default signal semantics when Neatlogs was alone.
  const hostOwnsSignal = process
    .listeners(signal)
    .some((listener) => listener !== _shutdownOnSignal);
  void shutdown(signal)
    .catch((error) => {
      logger.error(`Error shutting down after ${signal}: ${error}`);
    })
    .finally(() => {
      if (hostOwnsSignal || process.listenerCount(signal) > 0) return;
      try {
        process.kill(process.pid, signal);
      } catch {
        process.exitCode = signal === "SIGINT" ? 130 : 143;
      }
    });
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Resolve a non-empty workflow name; derive from process.argv when omitted.
 */
function _resolveWorkflowName(workflowName?: string): string {
  const provided = (workflowName ?? "").trim();
  if (provided) return provided;

  const argv1 = process.argv[1] ?? "";
  const base = path.basename(argv1, path.extname(argv1));
  const slug = base
    .replace(/[^a-zA-Z0-9_-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .toLowerCase();

  if (slug && !["node", "ts-node", "tsx", "npx", "-e"].includes(slug)) {
    return slug;
  }

  return "neatlogs-app";
}

const INIT_OPTION_KEYS = new Set<keyof InitOptions>([
  "apiKey",
  "experiments",
  "workflowName",
  "userId",
  "tags",
  "metadata",
  "debug",
  "disableExport",
  "diagnosticCapture",
  "doctorProbe",
  "doctorProbeExporter",
  "doctorProbeTimeoutMillis",
  "tracerProvider",
  "registerShutdownHandlers",
  "mask",
  "sampleRate",
  "captureLogs",
  "pii",
  "version",
  "endpoint",
  "batchSize",
  "flushInterval",
  "piiEnabled",
  "piiSpanTypes",
  "uploadAuthority",
]);

/** Explicit `endpoint` wins, then NEATLOGS_ENDPOINT, then the prod default. */
export function resolveInitEndpoint(endpoint: string | undefined): string {
  const explicit = (endpoint ?? "").trim();
  if (explicit) return explicit;
  const fromEnv = (process.env.NEATLOGS_ENDPOINT ?? "").trim();
  return fromEnv || DEFAULT_INGEST_ENDPOINT;
}

/**
 * Parse a resolved ingest endpoint down to its base origin. The endpoint must
 * be a bare base URL or an OTLP traces URL ending in /v1/traces. Any other
 * path would be silently dropped by origin parsing and misroute telemetry, so
 * it is rejected instead (same contract as the python SDK).
 */
export function resolveIngestBaseUrl(endpoint: string): string {
  const parsed = new URL(endpoint.trim());
  const path = parsed.pathname.replace(/\/+$/, "");
  if (path !== "" && path !== "/v1/traces") {
    throw new NeatlogsConfigurationError(
      "INVALID_ENDPOINT",
      "endpoint",
      "endpoint must be a base URL or an OTLP traces URL ending in /v1/traces.",
    );
  }
  return parsed.origin;
}

function validateInitOptions(options: InitOptions): void {
  const raw = options as Record<string, unknown>;
  if (Object.prototype.hasOwnProperty.call(raw, "instrumentations")) {
    throw new NeatlogsConfigurationError(
      "UNSUPPORTED_INSTRUMENTATIONS",
      "instrumentations",
      "The instrumentations option was removed. Use an explicit Neatlogs wrapper, handler, hook, processor, or plugin for the library instance.",
    );
  }
  const unknown = Object.keys(raw).find(
    (key) => !INIT_OPTION_KEYS.has(key as keyof InitOptions),
  );
  if (unknown) {
    throw new NeatlogsConfigurationError(
      "UNKNOWN_INIT_OPTION",
      unknown,
      `Unknown Neatlogs init option: ${unknown}`,
    );
  }
  if (
    options.sampleRate !== undefined &&
    (!Number.isFinite(options.sampleRate) ||
      options.sampleRate < 0 ||
      options.sampleRate > 1)
  ) {
    throw new RangeError("sampleRate must be a finite number between 0 and 1.");
  }
  if (
    options.uploadAuthority !== undefined &&
    typeof options.uploadAuthority !== "boolean" &&
    !isUploadAuthority(options.uploadAuthority)
  ) {
    throw new TypeError(
      "uploadAuthority must be a boolean or an UploadAuthority implementation",
    );
  }
  if (options.doctorProbeExporter !== undefined && options.doctorProbe !== true) {
    throw new TypeError("doctorProbeExporter requires doctorProbe: true");
  }
  if (options.doctorProbeTimeoutMillis !== undefined && options.doctorProbe !== true) {
    throw new TypeError("doctorProbeTimeoutMillis requires doctorProbe: true");
  }
  if (
    options.doctorProbeTimeoutMillis !== undefined &&
    (!Number.isFinite(options.doctorProbeTimeoutMillis) || options.doctorProbeTimeoutMillis <= 0)
  ) {
    throw new RangeError("doctorProbeTimeoutMillis must be a positive finite number");
  }
  if (options.doctorProbe === true && options.diagnosticCapture !== true) {
    throw new TypeError("doctorProbe requires diagnosticCapture: true");
  }
  if (
    options.diagnosticCapture === true &&
    options.disableExport !== true &&
    options.doctorProbe !== true
  ) {
    throw new TypeError(
      "diagnosticCapture requires disableExport: true or doctorProbe: true",
    );
  }
}

function stableValue(
  value: unknown,
  ancestors = new WeakSet<object>(),
  location = "init options",
): string {
  if (value === null) return "null";
  if (value === undefined) return "undefined";
  if (typeof value === "string") return `string:${JSON.stringify(value)}`;
  if (typeof value === "boolean") return `boolean:${value}`;
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new TypeError(`${location} must contain only finite numbers`);
    }
    return `number:${Object.is(value, -0) ? "-0" : String(value)}`;
  }
  if (typeof value !== "object") {
    throw new TypeError(
      `${location} contains unsupported value type ${typeof value}`,
    );
  }
  if (ancestors.has(value)) {
    throw new TypeError(`${location} contains a circular reference`);
  }

  ancestors.add(value);
  try {
    if (Array.isArray(value)) {
      return `array:[${value
        .map((item, index) => stableValue(item, ancestors, `${location}[${index}]`))
        .join(",")}]`;
    }
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
      throw new TypeError(
        `${location} contains unsupported value ${prototype?.constructor?.name ?? "object"}`,
      );
    }
    const record = value as Record<string, unknown>;
    return `object:{${Object.keys(record)
      .sort((left, right) => left.localeCompare(right))
      .map(
        (key) =>
          `${JSON.stringify(key)}:${stableValue(record[key], ancestors, `${location}.${key}`)}`,
      )
      .join(",")}}`;
  } finally {
    ancestors.delete(value);
  }
}

/** Explicit non-empty `apiKey` wins, then NEATLOGS_API_KEY. */
export function resolveApiKey(apiKey: string | undefined): string {
  const explicit = (apiKey ?? "").trim();
  if (explicit) return explicit;
  return (process.env.NEATLOGS_API_KEY ?? "").trim();
}

function initIdentity(options: InitOptions): InitIdentity {
  const apiKey = resolveApiKey(options.apiKey);
  const apiKeyDigest = createHash("sha256").update(apiKey).digest("hex");
  const serialized = stableValue({
    apiKeyDigest,
    experiments: options.experiments ?? null,
    workflowName: _resolveWorkflowName(options.workflowName),
    userId: options.userId ?? null,
    tags: options.tags ?? [],
    metadata: options.metadata ?? {},
    debug: options.debug ?? false,
    disableExport:
      !!options.disableExport ||
      ["true", "1", "yes"].includes(
        (process.env.NEATLOGS_DISABLE_EXPORT ?? "").toLowerCase(),
      ),
    diagnosticCapture: options.diagnosticCapture ?? false,
    doctorProbe: options.doctorProbe ?? false,
    doctorProbeTimeoutMillis: options.doctorProbeTimeoutMillis ?? null,
    registerShutdownHandlers: options.registerShutdownHandlers ?? null,
    sampleRate: options.sampleRate ?? 1,
    captureLogs: options.captureLogs ?? false,
    pii: options.pii ?? null,
    version: options.version ?? null,
    endpoint: resolveInitEndpoint(options.endpoint),
    batchSize: options.batchSize ?? 100,
    flushInterval: options.flushInterval ?? 5,
    piiEnabled: options.piiEnabled ?? null,
    piiSpanTypes: options.piiSpanTypes ?? [],
    uploadsEnabled:
      typeof options.uploadAuthority === "boolean"
        ? options.uploadAuthority
        : isUploadAuthority(options.uploadAuthority)
          ? options.uploadAuthority.available
          : uploadsEnabledFromEnv(process.env.NEATLOGS_UPLOADS_ENABLED),
  });
  return {
    serialized,
    mask: options.mask,
    tracerProvider: options.tracerProvider,
    uploadAuthority: isUploadAuthority(options.uploadAuthority)
      ? options.uploadAuthority
      : undefined,
    doctorProbeExporter: options.doctorProbeExporter,
  };
}

function sameInitIdentity(
  left: InitIdentity | null,
  right: InitIdentity,
): boolean {
  return (
    left !== null &&
    left.serialized === right.serialized &&
    left.mask === right.mask &&
    left.tracerProvider === right.tracerProvider &&
    left.uploadAuthority === right.uploadAuthority &&
    left.doctorProbeExporter === right.doctorProbeExporter
  );
}

function conflictingInit(): Promise<void> {
  return Promise.reject(
    new NeatlogsConfigurationError(
      "CONFLICTING_INIT",
      "init",
      "Neatlogs is already initializing or running with different configuration. Shut it down before reinitializing.",
    ),
  );
}

// ---------------------------------------------------------------------------
// init()
// ---------------------------------------------------------------------------

/**
 * Initialise the Neatlogs SDK.
 *
 * Sets up OTel TracerProvider, MeterProvider, LoggerProvider, span processors,
 * and exporters. It returns a Promise so initialization remains lifecycle-safe
 * across concurrent callers and future asynchronous transports.
 */
export function init(options: InitOptions = {}): Promise<void> {
  let identity: InitIdentity;
  try {
    validateInitOptions(options);
    configureReplayFromEnvironment();
    identity = initIdentity(options);
  } catch (error) {
    return Promise.reject(error);
  }
  if (_lifecycleState === "initializing" && _initPromise) {
    return sameInitIdentity(_initIdentity, identity)
      ? _initPromise
      : conflictingInit();
  }
  if (_lifecycleState === "closing" && _shutdownPromise) {
    return _shutdownPromise.then(() => init(options));
  }
  if (_lifecycleState === "running" || _initialized) {
    if (sameInitIdentity(_initIdentity, identity)) {
      logger.warn("Neatlogs already initialized with the same configuration");
      return Promise.resolve();
    }
    return conflictingInit();
  }

  _lifecycleState = "initializing";
  _initIdentity = identity;
  const current = Promise.resolve().then(() => _performInit(options));
  _initPromise = current;
  void current.then(
    () => {
      if (_initPromise === current) _initPromise = null;
      if (_lifecycleState === "initializing") _lifecycleState = "running";
    },
    () => {
      if (_initPromise === current) _initPromise = null;
      if (_lifecycleState === "initializing") {
        closeSharedClient();
        setAbTestClient(undefined);
        _lifecycleState = "uninitialized";
        _initIdentity = null;
      }
    },
  );
  return current;
}

async function _performInit(options: InitOptions): Promise<void> {
  // A new runtime must not diagnose envelopes from a previous initialization.
  clearDoctorCapture();
  const sampleRate = options.sampleRate ?? 1.0;
  if (!Number.isFinite(sampleRate) || sampleRate < 0 || sampleRate > 1) {
    throw new RangeError("sampleRate must be a finite number between 0 and 1.");
  }
  if (options.tracerProvider && options.sampleRate !== undefined) {
    throw new Error(
      "sampleRate cannot configure a caller-owned tracerProvider; configure its sampler directly.",
    );
  }
  _deliveryDiagnostics = new DeliveryDiagnostics();

  // 2. Resolve API key
  let resolvedKey = resolveApiKey(options.apiKey);

  // 3. Resolve disableExport
  let disableExportResolved =
    !!options.disableExport ||
    ["true", "1", "yes"].includes(
      (process.env.NEATLOGS_DISABLE_EXPORT ?? "").toLowerCase(),
    );

  if (!resolvedKey) {
    disableExportResolved = true;
    resolvedKey = "disabled";
    if (options.debug) {
      logger.warn(
        "No NEATLOGS_API_KEY set; HTTP export disabled. " +
          "Set NEATLOGS_API_KEY (or pass apiKey) to send spans to the backend.",
      );
    }
  }
  _effectiveSampleRate = sampleRate;
  _exportEnabled = !disableExportResolved;
  _queueMaxSize = 2_048;

  // 4. Debug mode
  if (options.debug) {
    enableDebugLogging();
  }
  _debugMode = options.debug ?? false;

  // 5. Resolve workflow name
  const resolvedWorkflowName = _resolveWorkflowName(options.workflowName);

  // 6. Parse base URL from endpoint
  const endpoint = resolveInitEndpoint(options.endpoint);
  const baseUrl = resolveIngestBaseUrl(endpoint);
  const uploadAuthority = disableExportResolved
    ? new DisabledUploadAuthority("export_disabled")
    : resolveUploadAuthority(
        options.uploadAuthority,
        process.env.NEATLOGS_UPLOADS_ENABLED,
        baseUrl,
        resolvedKey,
      );
  _deliveryDiagnostics.configureUploadAuthority(
    uploadAuthority.available,
    uploadAuthority.unavailableReason,
  );

  // 7. Set session config. Session & end-user identity are PER-REQUEST (set via
  // trace()/span() or identify()), never on init() — only the operator userId
  // and workflow/transport config live here.
  _setSessionConfig({
    userId: options.userId,
    workflowName: resolvedWorkflowName,
    _apiKey: resolvedKey,
    _baseUrl: baseUrl,
  });

  // 8b. Initialize shared PromptClient so module-level helpers work after init()
  if (resolvedKey && resolvedKey !== "disabled") {
    setSharedClient(new PromptClient({ baseUrl, apiKey: resolvedKey }));
  }
  if (options.experiments && !disableExportResolved) {
    const experiments = new AbTestClient({
      baseUrl: options.experiments.baseUrl ?? baseUrl,
      apiKey: resolvedKey,
      keys: options.experiments.keys,
    });
    setAbTestClient(experiments);
    await experiments.start();
  }

  // 9. Build OTel Resource
  const resourceAttrs: Record<string, string | number | boolean> = {
    [ATTR_SERVICE_NAME]: options.workflowName || "neatlogs-app",
    "service.version": __version__,
    "neatlogs.workflow_name": resolvedWorkflowName,
  };
  if (options.doctorProbe) {
    resourceAttrs['neatlogs.doctor'] = true;
    resourceAttrs['neatlogs.doctor.version'] = 'v1';
    resourceAttrs['telemetry.sdk.language'] = 'typescript';
    resourceAttrs['telemetry.sdk.version'] = __version__;
  }
  addVerificationMarkerResourceAttribute(resourceAttrs);
  // Operator identity only — whoever RUNS the SDK. Session & end-user identity
  // are per-request (trace()/span()/identify()), never resource attributes.
  if (options.userId) resourceAttrs["user.id"] = options.userId;

  const tags = options.tags;
  if (tags !== undefined) {
    if (!Array.isArray(tags) || !tags.every((t) => typeof t === "string")) {
      throw new Error("tags must be a list of strings");
    }
    resourceAttrs["neatlogs.tags"] = tags.join(",");
  }

  if (options.pii !== undefined) {
    resourceAttrs["neatlogs.pii.enabled"] =
      options.pii === false ? "false" : "true";
  }

  if (options.piiSpanTypes !== undefined) {
    resourceAttrs["neatlogs.pii.span_types"] = options.piiSpanTypes.join(",");
  }

  const resource = new Resource(resourceAttrs);

  // 10. Create or adopt a PRIVATE TracerProvider. Neatlogs never reads,
  // registers onto, or shuts down a foreign global provider, so Datadog,
  // Braintrust, and other co-tenants can neither export nor parent our spans
  // (and vice versa).
  const provider =
    options.tracerProvider ??
    new NodeTracerProvider({
      resource,
      spanLimits: spanLimitsForCaptureEverything(),
      sampler: new ParentBasedSampler({
        root: new TraceIdRatioBasedSampler(sampleRate),
      }),
    });
  _ownsTracerProvider = options.tracerProvider === undefined;

  // 11. Add NeatlogsSpanProcessor
  _spanProcessor = new NeatlogsSpanProcessor({
    debug: options.debug ?? false,
    mask: options.mask,
    emitCompletionMarkers: false,
    ownAllSpans: _ownsTracerProvider,
    mediaUploadsAvailable: uploadAuthority.available,
    mediaUploadsUnavailableReason: uploadAuthority.unavailableReason,
  });
  provider.addSpanProcessor(_spanProcessor);

  // 12. Add BatchSpanProcessor + exporter. Local Doctor uses the exact same
  // final masking/filtering boundary with a successful in-memory sink, so it
  // can inspect export-ready bytes without credentials or network access.
  if (!disableExportResolved || options.diagnosticCapture) {
    const tracesEndpoint = endpoint.endsWith("/v1/traces")
      ? endpoint
      : `${baseUrl}/v1/traces`;

    const diagnosticSink: SpanExporter = {
      export(_spans, resultCallback) {
        resultCallback({ code: ExportResultCode.SUCCESS });
      },
      async shutdown() {},
      async forceFlush() {},
    };

    const transportExporter = disableExportResolved
      ? diagnosticSink
      : (options.doctorProbeExporter ?? new ByteLimitedSpanExporter(
          new OTLPTraceExporter({
            url: tracesEndpoint,
            ...(options.doctorProbeTimeoutMillis !== undefined
              ? { timeoutMillis: options.doctorProbeTimeoutMillis }
              : {}),
            headers: {
              "x-api-key": resolvedKey,
              ...(options.doctorProbe ? { "x-neatlogs-doctor": "v1" } : {}),
            },
            compression: CompressionAlgorithm.GZIP,
          }),
          undefined,
          _deliveryDiagnostics,
          uploadAuthority,
        ));
    const otlpExporter = new FilteringExporter(
      transportExporter,
      _deliveryDiagnostics,
      uploadAuthority,
      options.diagnosticCapture ? capturePreparedSpans : undefined,
    );

    const batchSize = options.batchSize ?? 100;
    const batchProcessor = new ObservableBatchSpanProcessor(
      otlpExporter,
      {
        maxExportBatchSize: batchSize,
        maxQueueSize: exportQueueCapacity(batchSize),
        scheduledDelayMillis: (options.flushInterval ?? 5) * 1000,
      },
      _deliveryDiagnostics,
    );
    const completionProcessor = new CompletionMarkerSpanProcessor(
      _spanProcessor,
      provider.getTracer("neatlogs.internal"),
    );
    provider.addSpanProcessor(batchProcessor);
    provider.addSpanProcessor(completionProcessor);
    _transportSpanProcessors = [batchProcessor, completionProcessor];
    _completionProcessor = completionProcessor;

    if (options.debug && !options.diagnosticCapture) {
      logger.debug(`OTLP trace exporter configured: ${tracesEndpoint}`);
    } else if (options.debug) {
      logger.debug('Local Doctor diagnostic capture configured (network disabled)');
    }
  } else if (options.debug) {
    logger.debug("Export disabled — spans will not be sent to backend");
  }

  // 13. Store privately. Neatlogs never registers an OTel provider globally.
  _tracerProvider = provider;
  _setNeatlogsProvider(provider);
  if (!disableExportResolved && _ownsTracerProvider) registerCaptureProvider(provider, _spanProcessor, sampleRate, flush);

  if (options.debug) {
    logger.debug("Neatlogs tracer provider initialized");
  }

  // 14. Never replace the process-global meter provider.
  if (options.debug) {
    logger.debug("Global meter provider left unchanged");
  }

  // 15. Set up LoggerProvider (if captureLogs)
  const captureLogs = options.captureLogs ?? false;
  if (captureLogs) {
    const logRecordProcessors: ObservableBatchLogRecordProcessor[] = [];

    // OTLP log export to /v1/logs (same pattern as Python SDK)
    if (!disableExportResolved) {
      const logsEndpoint = endpoint.endsWith("/v1/logs")
        ? endpoint
        : `${baseUrl}/v1/logs`;
      const otlpLogExporter = new OTLPLogExporter({
        url: logsEndpoint,
        headers: resolvedKey ? { "x-api-key": resolvedKey } : undefined,
        compression: CompressionAlgorithm.GZIP,
      });
      const batchSize = options.batchSize ?? 100;
      logRecordProcessors.push(
        new ObservableBatchLogRecordProcessor(
          new MaskingLogExporter(
            otlpLogExporter,
            options.mask,
            undefined,
            _deliveryDiagnostics,
          ),
          {
            maxExportBatchSize: batchSize,
            maxQueueSize: exportQueueCapacity(batchSize),
            scheduledDelayMillis: (options.flushInterval ?? 5) * 1000,
          },
          _deliveryDiagnostics,
        ),
      );
      if (options.debug) {
        logger.debug(`OTLP log exporter configured: ${logsEndpoint}`);
      }
    }

    // sdk-logs >= 0.200 configures processors in the constructor and removed
    // addLogRecordProcessor(). Retain the older path so Neatlogs stays
    // compatible with applications that still resolve sdk-logs 0.57.
    const supportsDynamicProcessors =
      typeof LoggerProvider.prototype.addLogRecordProcessor === "function";
    _logProvider = supportsDynamicProcessors
      ? new LoggerProvider({ resource })
      : new LoggerProvider({
          resource,
          processors: logRecordProcessors,
        } as ConstructorParameters<typeof LoggerProvider>[0]);
    if (supportsDynamicProcessors) {
      for (const processor of logRecordProcessors) {
        _logProvider.addLogRecordProcessor(processor);
      }
    }

    // Wire the OTel logger for neatlogs.log() function
    const otelLogger = _logProvider.getLogger("neatlogs");
    _setOtelLogger(otelLogger, options.debug ?? false);

    if (options.debug) {
      logger.debug(
        `Neatlogs log capture enabled (endpoint: ${baseUrl}/v1/logs)`,
      );
    }
  } else if (options.debug) {
    logger.debug("Log capture disabled (pass captureLogs: true to enable)");
  }

  // 16. Register shutdown handlers. Default to flushing on exit whenever we own
  // the private provider so standalone scripts drain their spans;
  // only a caller-supplied provider defaults off (its owner controls shutdown).
  const registerShutdownHandlers =
    options.registerShutdownHandlers ?? _ownsTracerProvider;
  if (registerShutdownHandlers && !_sigHandlersRegistered) {
    process.on('beforeExit', _shutdownBeforeExit);
    // Run before pre-existing one-shot host listeners so ownership is observed
    // before EventEmitter removes them for their invocation.
    process.prependListener('SIGTERM', _shutdownOnSignal);
    process.prependListener('SIGINT', _shutdownOnSignal);
    _sigHandlersRegistered = true;
  }

  // 17. Mark as initialised
  _initialized = true;

  if (options.debug) {
    logger.info("Neatlogs SDK initialized successfully");
    logger.info(`Endpoint: ${endpoint}`);
    logger.info(`Workflow: ${resolvedWorkflowName}`);
    logger.info(`User: ${options.userId ?? "(none)"}`);
    logger.info(`Tags: ${tags ?? []}`);
    logger.info(`Sample Rate: ${sampleRate}`);
  }
}

// ---------------------------------------------------------------------------
// flush()
// ---------------------------------------------------------------------------

/**
 * Flush all pending spans, metrics, and log records.
 *
 * @returns true if all providers flushed successfully
 */
export async function flush(): Promise<boolean> {
  let success = true;

  // Flush regardless of ownership: our processors are attached to the provider
  // (owned or caller-supplied), so draining them is always ours to do. Shutdown
  // is what's ownership-gated — we never shut down a provider we didn't create.
  // Logs must drain before spans: the trace flush can export the root and its
  // completion marker, which may make the backend finalize the trace.
  if (_logProvider) {
    try {
      logger.debug("Flushing log provider...");
      await _logProvider.forceFlush();
      logger.debug("Log provider flushed successfully");
    } catch (e) {
      logger.error(`Error flushing logs: ${e}`);
      success = false;
    }
  }

  try {
    await _spanProcessor?.forceFlush();
    await _completionProcessor?.forceFlush();
  } catch (e) {
    logger.error(`Error completing span masking/finalization: ${e}`);
    success = false;
  }

  if (_tracerProvider) {
    try {
      logger.debug("Flushing tracer provider...");
      await _tracerProvider.forceFlush();
      logger.debug("Tracer provider flushed successfully");
    } catch (e) {
      logger.error(`Error flushing spans: ${e}`);
      success = false;
    }
  }

  const experiments = getAbTestClient();
  if (experiments && !(await experiments.flush())) success = false;
  return success;
}

/**
 * Flush the module-level SDK pipeline and every live Neatlogs `Client` in this
 * process. Foreign OpenTelemetry providers are never discovered or flushed.
 */
export interface FlushOutcome {
  pipeline: string;
  success: boolean;
  timedOut: boolean;
  error?: string;
}

export interface FlushAllResult {
  success: boolean;
  outcomes: FlushOutcome[];
}

export async function flushAllDetailed(
  timeoutMs = 30_000,
): Promise<FlushAllResult> {
  const deadline = Date.now() + Math.max(0, timeoutMs);
  const clients = getRegisteredClients();
  const operations: Array<{ pipeline: string; run: () => Promise<boolean> }> =
    [];
  if (_initialized) operations.push({ pipeline: "default", run: flush });
  clients.forEach((client, index) =>
    operations.push({
      pipeline: `client:${client.workflowName ?? "anonymous"}:${index}`,
      run: () => client.flush(),
    }),
  );

  const outcomes = await Promise.all(
    operations.map(async ({ pipeline, run }): Promise<FlushOutcome> => {
      const remaining = Math.max(0, deadline - Date.now());
      let timer: ReturnType<typeof setTimeout> | undefined;
      try {
        const timedOut = new Promise<never>((_resolve, reject) => {
          timer = setTimeout(
            () => reject(new Error("NEATLOGS_FLUSH_TIMEOUT")),
            remaining,
          );
        });
        const success = await Promise.race([
          Promise.resolve().then(run),
          timedOut,
        ]);
        return { pipeline, success: success === true, timedOut: false };
      } catch (error) {
        const timedOut =
          error instanceof Error && error.message === "NEATLOGS_FLUSH_TIMEOUT";
        return {
          pipeline,
          success: false,
          timedOut,
          error: timedOut
            ? "timeout"
            : error instanceof Error
              ? error.name
              : "Error",
        };
      } finally {
        if (timer !== undefined) clearTimeout(timer);
      }
    }),
  );
  return { success: outcomes.every((outcome) => outcome.success), outcomes };
}

/** Preserve the original aggregate boolean contract for shutdown callers. */
export async function flushAll(timeoutMs = 30_000): Promise<boolean> {
  return (await flushAllDetailed(timeoutMs)).success;
}

/** Snapshot bounded-queue, masking, and final-export loss counters. */
export function getDeliveryDiagnostics(): DeliveryDiagnosticsSnapshot {
  return _deliveryDiagnostics.snapshot();
}

// ---------------------------------------------------------------------------
// shutdown()
// ---------------------------------------------------------------------------

/**
 * Shutdown the SDK and flush all pending spans, metrics, and log records.
 *
 * Resets all module-level state so `init()` can be called again.
 *
 * @returns true if all providers shut down successfully
 */
export function shutdown(
  terminationReason = "shutdown",
  timeoutMs = 30_000,
): Promise<boolean> {
  if (_shutdownPromise) return _shutdownPromise;

  const initialization =
    _lifecycleState === "initializing" ? _initPromise : null;
  _lifecycleState = "closing";
  _completionProcessor?.beginShutdown();
  _spanProcessor?.beginShutdown(terminationReason);
  _setOtelLogger(null, false);

  // Defer all shutdown work until the shared promise has been installed.
  // endActiveSpans() synchronously invokes onEnd(), and a host processor may
  // re-enter shutdown() from there.
  const currentShutdown = Promise.resolve().then(async () => {
    if (initialization) {
      try {
        await initialization;
      } catch {
        // Teardown any partial state left by a failed initialization.
      }
    }
    return _performShutdown(terminationReason, timeoutMs);
  });
  _shutdownPromise = currentShutdown;
  const clearCurrentShutdown = () => {
    if (_shutdownPromise === currentShutdown) {
      _shutdownPromise = null;
      _lifecycleState = "uninitialized";
    }
  };
  void currentShutdown.then(clearCurrentShutdown, clearCurrentShutdown);
  return currentShutdown;
}

async function _performShutdown(
  terminationReason: string,
  timeoutMs: number,
): Promise<boolean> {
  // Remove signal handlers
  if (_sigHandlersRegistered) {
    process.removeListener("beforeExit", _shutdownBeforeExit);
    process.removeListener("SIGTERM", _shutdownOnSignal);
    process.removeListener("SIGINT", _shutdownOnSignal);
    _sigHandlersRegistered = false;
  }

  let success = true;
  const deadline = Date.now() + Math.max(0, timeoutMs);
  const logProvider = _logProvider;
  const spanProcessor = _spanProcessor;
  const completionProcessor = _completionProcessor;
  const tracerProvider = _tracerProvider;
  const ownsTracerProvider = _ownsTracerProvider;
  const transportSpanProcessors = [..._transportSpanProcessors];

  const attempt = async (
    label: string,
    operation: () => unknown | PromiseLike<unknown>,
  ): Promise<void> => {
    const outcome = await runByDeadline(operation, deadline);
    if (!outcome.completed) {
      logger.error(
        `${label} failed or exceeded the Neatlogs shutdown deadline`,
      );
      success = false;
    }
  };

  // Clear the Mastra bridge cache so a subsequent init() rebinds to the fresh
  // provider instead of a stale one.
  _resetMastraCache();

  // Prompt CRUD has an independent transport and must never affect telemetry
  // flush success. Closing it here only aborts its own refreshes and cache.
  closeSharedClient();

  // LOG records must drain before trace completion. The trace provider carries
  // neatlogs.trace.complete, which can trigger server-side finalization.
  if (logProvider) {
    logger.debug("Shutting down log provider...");
    await attempt("Log provider shutdown", () => logProvider.shutdown());
  }

  // Ending the root creates the trace-completion marker. Do this only after
  // buffered logs have drained so finalization cannot overtake them.
  if (spanProcessor) {
    const ended = spanProcessor.endActiveSpans(terminationReason);
    if (ended > 0) {
      logger.info(
        `Ended ${ended} active Neatlogs span(s) during ${terminationReason}`,
      );
    }
  }
  completionProcessor?.emitDeferred();
  if (spanProcessor) {
    await attempt("Span masking/finalization", () =>
      spanProcessor.forceFlush(),
    );
  }
  if (completionProcessor) {
    await attempt("Completion processor flush", () =>
      completionProcessor.forceFlush(),
    );
  }

  // Only shut down a provider we created. A caller-supplied provider is flushed
  // (above / in flush()) but never shut down — its owner controls its lifecycle,
  // per the tracerProvider contract in InitOptions. Flush it here so a caller
  // that shuts down right after us doesn't lose our still-buffered spans.
  if (tracerProvider && ownsTracerProvider) {
    logger.debug("Shutting down tracer provider...");
    await attempt("Tracer provider shutdown", () => tracerProvider.shutdown());
  } else if (tracerProvider) {
    logger.debug(
      "Flushing caller-owned tracer provider (not shutting down)...",
    );
    await attempt("Caller-owned tracer provider flush", () =>
      tracerProvider.forceFlush(),
    );
    // The provider is caller-owned, but these processors/exporters are ours.
    // SDKs cannot detach processors in OTel JS 1.x; shut every instance down
    // independently so one exporter failure cannot leave another active.
    for (const processor of transportSpanProcessors.reverse()) {
      await attempt("Neatlogs transport shutdown", () => processor.shutdown());
    }
    if (spanProcessor) {
      await attempt("Neatlogs span processor shutdown", () =>
        spanProcessor.shutdown(),
      );
    }
  }

  const experiments = getAbTestClient();
  if (experiments) {
    await attempt('A/B exposures', async () => {
      if (!(await experiments.flush())) success = false;
    });
    setAbTestClient(undefined);
  }

  // Reset all module-level state
  _initialized = false;
  _tracerProvider = null;
  _ownsTracerProvider = false;
  _setNeatlogsProvider(null);
  discardPendingMediaOwner();
  _logProvider = null;
  _spanProcessor = null;
  _transportSpanProcessors = [];
  _completionProcessor = null;
  _debugMode = false;
  _effectiveSampleRate = 1;
  _exportEnabled = false;
  _queueMaxSize = 2_048;
  _signalShutdownStarted = false;
  _initIdentity = null;
  clearDoctorCapture();

  // Reset session config
  _setSessionConfig({});

  logger.info("Neatlogs SDK shutdown complete");
  return success;
}

// ---------------------------------------------------------------------------
// getTracerProvider()
// ---------------------------------------------------------------------------

/**
 * Return the active TracerProvider. Throws if init() has not been called.
 */
export function getTracerProvider(): BasicTracerProvider {
  if (!_tracerProvider) {
    throw new Error(
      "Neatlogs is not initialized. Call init() before accessing the TracerProvider.",
    );
  }
  return _tracerProvider;
}

// ---------------------------------------------------------------------------
// isDebugEnabled()
// ---------------------------------------------------------------------------

/**
 * Return true if neatlogs was initialised with `debug: true`.
 */
export function isDebugEnabled(): boolean {
  return _debugMode;
}

// ---------------------------------------------------------------------------
// Re-export getSessionConfig from context
// ---------------------------------------------------------------------------

export { getSessionConfig } from "./core/context.js";

/** @internal Immutable, credential-free state used by the read-only doctor. */
export function _doctorRuntimeSnapshot(): Readonly<{
  state: LifecycleState;
  initialized: boolean;
  ownsTracerProvider: boolean;
  exportHealth: { droppedSpans: number; exportFailures: number } | null;
  effectiveSampler: string;
  exportEnabled: boolean;
  queueMaxSize: number;
}> {
  return {
    state: _lifecycleState,
    initialized: _initialized,
    ownsTracerProvider: _ownsTracerProvider,
    exportHealth: {
      droppedSpans:
        _deliveryDiagnostics.snapshot().spanQueueDrops +
        _deliveryDiagnostics.snapshot().frameworkSpanDrops +
        _deliveryDiagnostics.snapshot().maskedSpanDrops,
      exportFailures: _deliveryDiagnostics.snapshot().spanExportFailures,
    },
    effectiveSampler: `parentbased_traceidratio:${_effectiveSampleRate}`,
    exportEnabled: _exportEnabled,
    queueMaxSize: _queueMaxSize,
  };
}
