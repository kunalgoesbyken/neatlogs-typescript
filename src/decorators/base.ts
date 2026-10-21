/**
 * Core span decoration utilities.
 * Provides the low-level wrapping logic used by span() and Span().
 */

import {
  ROOT_CONTEXT,
  SpanStatusCode,
  type Span as OtelSpan,
} from "@opentelemetry/api";
import { getLogger } from "../core/logger.js";
import { isAbortError, markCancelled } from "../core/span-outcome.js";
import { registerMask } from "../core/mask.js";
import { applyEndUserAttributes, isRootSpan } from "../core/end-user.js";
import { applySessionAttributes } from "../core/session.js";
import {
  captureMedia,
  captureMediaWithIndex,
  sanitizeMediaPayload,
} from "../core/media.js";
import {
  getNeatlogsTracer,
  getNeatlogsParentContext,
  withNeatlogsSpan,
} from "../core/provider.js";
import {
  bindSpanToAutoRoot,
  maybeOpenAutoRoot,
} from "../core/auto-root.js";
import type { SpanOptions, MaskFunction } from "../types.js";
import { lookupReplay } from "../replay/session.js";
import {
  DEFAULT_MAX_SEMANTIC_STREAM_EVENTS,
  DEFAULT_MAX_STREAM_CAPTURE_BYTES,
  utf8ByteLength,
} from "../constants.js";

const logger = getLogger();
const TRACER_NAME = "neatlogs";
const STREAM_CAPTURE_TRUNCATED = "[TRUNCATED_STREAM_OUTPUT]";

/** Safely serialize any value to JSON string. */
export function safeJsonDumps(value: any): string {
  try {
    return JSON.stringify(value, (_key, val) => {
      if (typeof val === "bigint") return val.toString();
      if (val instanceof Error)
        return { message: val.message, name: val.name, stack: val.stack };
      if (typeof val === "function")
        return `[Function: ${val.name || "anonymous"}]`;
      return val;
    });
  } catch {
    return String(value);
  }
}

/**
 * Serialize an object for span attributes. Handles toJSON(), plain objects,
 * primitives. A reference cycle is replaced with "[Circular]" instead of
 * recursing until the stack overflows, so tracing a request, an ORM entity or
 * any self-referencing value can never throw into the caller's code.
 */
export function serializeObj(obj: any): any {
  return serializeInner(obj, new WeakSet());
}

function serializeInner(obj: any, ancestors: WeakSet<object>): any {
  if (obj === null || obj === undefined) return obj;
  // Preserve binary views until media capture can classify and sanitize them.
  // JSON.stringify retains the previous numeric-object representation for
  // ordinary, non-media Uint8Array values.
  if (obj instanceof Uint8Array) return obj;
  if (
    typeof obj === "string" ||
    typeof obj === "number" ||
    typeof obj === "boolean"
  )
    return obj;
  if (typeof obj.toJSON === "function") return obj.toJSON();
  if (typeof obj === "object") {
    if (ancestors.has(obj)) return "[Circular]";
    ancestors.add(obj);
    try {
      if (Array.isArray(obj)) return obj.map((item) => serializeInner(item, ancestors));
      const result: Record<string, any> = {};
      for (const [key, value] of Object.entries(obj)) {
        result[key] = serializeInner(value, ancestors);
      }
      return result;
    } finally {
      ancestors.delete(obj);
    }
  }
  return String(obj);
}

/** Set common span attributes from options. */
export function setCommonSpanAttrs(span: OtelSpan, opts: SpanOptions): void {
  if (opts.kind) {
    span.setAttribute("openinference.span.kind", opts.kind);
  }
  if (opts.internal) {
    span.setAttribute("neatlogs.internal", true);
  }
  if (opts.description) {
    span.setAttribute("neatlogs.description", opts.description);
  }
  if (opts.mask) {
    const maskId = registerMask(opts.mask);
    span.setAttribute("neatlogs.mask_id", maskId);
  }
}

export interface DecorateSpanOptions extends SpanOptions {
  /** Override the span name (defaults to function name). */
  spanName?: string;
  /** Post-process callback after the function returns. */
  postprocessResult?: (
    span: OtelSpan,
    result: any,
    boundInputs: Record<string, any>,
  ) => void;
}

function isPromiseLike(value: unknown): value is PromiseLike<unknown> {
  return (
    value !== null &&
    (typeof value === "object" || typeof value === "function") &&
    typeof (value as { then?: unknown }).then === "function"
  );
}

function isAsyncIterable(value: unknown): value is AsyncIterable<unknown> {
  return (
    value !== null &&
    (typeof value === "object" || typeof value === "function") &&
    typeof (value as { [Symbol.asyncIterator]?: unknown })[
      Symbol.asyncIterator
    ] === "function"
  );
}

function isSyncIterator(
  value: unknown,
): value is Iterator<unknown> & Iterable<unknown> {
  return (
    value !== null &&
    (typeof value === "object" || typeof value === "function") &&
    typeof (value as { next?: unknown }).next === "function" &&
    typeof (value as { [Symbol.iterator]?: unknown })[Symbol.iterator] ===
      "function"
  );
}

function isReadableStreamLike(value: unknown): value is {
  getReader: (...args: any[]) => any;
  cancel: (...args: any[]) => Promise<unknown>;
} {
  return (
    value !== null &&
    typeof value === "object" &&
    typeof (value as { getReader?: unknown }).getReader === "function" &&
    typeof (value as { cancel?: unknown }).cancel === "function"
  );
}

/**
 * Core wrapper factory that creates an instrumented version of a function.
 * This is the low-level building block used by span() and Span().
 */
export function decorateSpan<TArgs extends any[], TReturn>(
  opts: DecorateSpanOptions,
  fn: (...args: TArgs) => TReturn,
): (...args: TArgs) => TReturn {
  const spanName = opts.spanName ?? opts.name ?? (fn.name || "anonymous");
  const captureInput = opts.captureInput !== false;
  const captureOutput = opts.captureOutput !== false;
  const mediaPrefix = (direction: "input" | "output"): string => {
    const kind = String(opts.kind ?? "span").trim().toLowerCase() || "span";
    return `neatlogs.${kind}.${direction}`;
  };

  const wrapped = (...args: TArgs): any => {
    // Resolve the tracer at CALL time, not definition time: a decorator applied
    // before init() would otherwise capture the no-op global provider forever
    // and emit zero spans once the private provider is configured.
    const tracer = getNeatlogsTracer(TRACER_NAME);
    // End-user belongs to the trace root only. Capture root status BEFORE the
    // span is created (once it's active, it would be the "current" span).
    const isRoot = isRootSpan();
    // Parent from our private span store; a root decorated function anchors a
    // fresh trace under ROOT_CONTEXT.
    const parentContext = isRoot ? ROOT_CONTEXT : getNeatlogsParentContext();
    const autoRoot = maybeOpenAutoRoot(
      tracer,
      String(opts.kind ?? ""),
      parentContext,
    );
    if (autoRoot.root) {
      applySessionAttributes(autoRoot.root, opts.sessionId, true, {
        parentSessionId: opts.parentSessionId,
        sessionFeatureName: opts.sessionFeatureName,
        sessionEntryPoint: opts.sessionEntryPoint,
      });
      applyEndUserAttributes(
        autoRoot.root,
        opts.endUserId,
        opts.endUserMetadata,
        true,
      );
    }
    const span = bindSpanToAutoRoot(
      tracer.startSpan(spanName, {}, autoRoot.ctx),
      autoRoot.root,
    );
    return withNeatlogsSpan(span, () => {
      let finished = false;
      let streamInterrupted = false;
      const streamChunks: any[] = [];
      let totalStreamChunks = 0;
      let retainedStreamBytes = 0;
      let droppedStreamChunks = 0;
      let streamMediaCount = 0;

      const captureResult = (result: any): void => {
        if (captureOutput) {
          try {
            const serialized = serializeObj(result);
            const captured = captureMedia(
              span,
              mediaPrefix("output"),
              serialized,
              "output",
            );
            span.setAttribute(
              "output.value",
              safeJsonDumps(captured),
            );
          } catch (err) {
            logger.debug(`Failed to capture output: ${err}`);
          }
        }

        if (opts.postprocessResult) {
          try {
            const boundInputs = _extractBoundInputs(fn, args);
            opts.postprocessResult(span, result, boundInputs);
          } catch (err) {
            logger.debug(`Postprocess failed: ${err}`);
          }
        }
      };

      const finishSuccess = (result: any): void => {
        if (finished) return;
        finished = true;
        captureResult(result);
        span.setStatus({ code: SpanStatusCode.OK });
        span.end();
      };

      const setStreamCounts = (): void => {
        span.setAttribute("neatlogs.stream.chunk_count", totalStreamChunks);
        if (totalStreamChunks > DEFAULT_MAX_SEMANTIC_STREAM_EVENTS) {
          span.setAttribute(
            "neatlogs.stream.events_dropped",
            totalStreamChunks - DEFAULT_MAX_SEMANTIC_STREAM_EVENTS,
          );
        }
        if (droppedStreamChunks > 0) {
          span.setAttribute("neatlogs.stream.output_truncated", true);
          span.setAttribute(
            "neatlogs.stream.chunks_not_captured",
            droppedStreamChunks,
          );
          span.setAttribute(
            "neatlogs.stream.retained_bytes",
            retainedStreamBytes,
          );
        }
      };

      const streamCaptureValue = (): any[] =>
        droppedStreamChunks > 0
          ? [
              ...streamChunks,
              {
                neatlogs_stream_capture: STREAM_CAPTURE_TRUNCATED,
                chunks_not_captured: droppedStreamChunks,
              },
            ]
          : streamChunks;

      const finishError = (error: any): void => {
        if (finished) return;
        finished = true;
        if (isAbortError(error)) {
          if (totalStreamChunks > 0) {
            setStreamCounts();
            captureResult(streamCaptureValue());
          }
          markCancelled(span);
          span.end();
          return;
        }
        span.setStatus({
          code: SpanStatusCode.ERROR,
          message: error?.message ?? String(error),
        });
        span.recordException(
          error instanceof Error ? error : new Error(String(error)),
        );
        span.end();
      };

      const recordChunk = (chunk: any): void => {
        const index = totalStreamChunks++;
        if (index >= DEFAULT_MAX_SEMANTIC_STREAM_EVENTS) {
          droppedStreamChunks += 1;
          return;
        }

        let serializedValue: unknown;
        let serialized: string;
        try {
          serializedValue = serializeObj(chunk);
          serialized = safeJsonDumps(
            sanitizeMediaPayload(serializedValue, "output"),
          );
        } catch {
          droppedStreamChunks += 1;
          return;
        }
        const encodedBytes = utf8ByteLength(
          serialized,
          DEFAULT_MAX_STREAM_CAPTURE_BYTES - retainedStreamBytes,
        );
        if (
          retainedStreamBytes + encodedBytes >
          DEFAULT_MAX_STREAM_CAPTURE_BYTES
        ) {
          droppedStreamChunks += 1;
          return;
        }

        const captured = captureMediaWithIndex(
          span,
          mediaPrefix("output"),
          serializedValue,
          "output",
          streamMediaCount,
        );
        streamMediaCount += captured.count;
        streamChunks.push(captured.value);
        retainedStreamBytes += encodedBytes;
        const summary: Record<string, unknown> = {
          value_type: Array.isArray(chunk) ? "array" : typeof chunk,
          encoded_bytes: encodedBytes,
        };
        if (Array.isArray(chunk)) summary.items = chunk.length;
        else if (chunk && typeof chunk === "object")
          summary.keys = Object.keys(chunk).sort();
        span.addEvent("neatlogs.stream.chunk", {
          "neatlogs.stream.chunk.index": index,
          "neatlogs.stream.chunk.summary": safeJsonDumps(summary),
        });
      };

      const finishStream = (cancelled = false): void => {
        setStreamCounts();
        if (!cancelled) {
          finishSuccess(streamCaptureValue());
          return;
        }
        if (finished) return;
        finished = true;
        captureResult(streamCaptureValue());
        markCancelled(span);
        span.end();
      };

      const wrapAsyncIterator = (iterator: any): any => {
        let proxy: any;
        proxy = new Proxy(iterator, {
          get(target, property) {
            if (property === Symbol.asyncIterator) return () => proxy;
            if (property === "next") {
              return async (...nextArgs: any[]) => {
                try {
                  const item = await withNeatlogsSpan(span, () =>
                    target.next(...nextArgs),
                  );
                  if (item.done) finishStream(streamInterrupted);
                  else recordChunk(item.value);
                  return item;
                } catch (error) {
                  finishError(error);
                  throw error;
                }
              };
            }
            if (property === "return") {
              return async (...returnArgs: any[]) => {
                try {
                  streamInterrupted = true;
                  const item = target.return
                    ? await withNeatlogsSpan(span, () =>
                        target.return(...returnArgs),
                      )
                    : { done: true, value: returnArgs[0] };
                  if (item.done) finishStream(true);
                  return item;
                } catch (error) {
                  finishError(error);
                  throw error;
                }
              };
            }
            if (property === "throw") {
              return async (...throwArgs: any[]) => {
                try {
                  const item = target.throw
                    ? await withNeatlogsSpan(span, () =>
                        target.throw(...throwArgs),
                      )
                    : Promise.reject(throwArgs[0]);
                  if (item.done) finishStream();
                  else recordChunk(item.value);
                  return item;
                } catch (error) {
                  finishError(error);
                  throw error;
                }
              };
            }
            const value = Reflect.get(target, property, target);
            return typeof value === "function" ? value.bind(target) : value;
          },
        });
        return proxy;
      };

      const wrapAsyncIterable = (source: AsyncIterable<unknown>): any => {
        if (typeof (source as any).next === "function") {
          return wrapAsyncIterator(source);
        }
        return new Proxy(source as any, {
          get(target, property) {
            if (property === Symbol.asyncIterator) {
              return () => wrapAsyncIterator(target[Symbol.asyncIterator]());
            }
            const value = Reflect.get(target, property, target);
            return typeof value === "function" ? value.bind(target) : value;
          },
        });
      };

      const wrapSyncIterator = (
        iterator: Iterator<unknown> & Iterable<unknown>,
      ): any => {
        let proxy: any;
        proxy = new Proxy(iterator, {
          get(target, property) {
            if (property === Symbol.iterator) return () => proxy;
            if (property === "next") {
              return (...nextArgs: any[]) => {
                try {
                  const item = withNeatlogsSpan(span, () =>
                    (
                      target.next as (
                        ...args: any[]
                      ) => IteratorResult<unknown>
                    )(...nextArgs),
                  );
                  if (item.done) finishStream(streamInterrupted);
                  else recordChunk(item.value);
                  return item;
                } catch (error) {
                  finishError(error);
                  throw error;
                }
              };
            }
            if (property === "return") {
              return (...returnArgs: any[]) => {
                try {
                  streamInterrupted = true;
                  const item = target.return
                    ? withNeatlogsSpan(span, () =>
                        (
                          target.return as (
                            ...args: any[]
                          ) => IteratorResult<unknown>
                        )(...returnArgs),
                      )
                    : { done: true, value: returnArgs[0] };
                  if (item.done) finishStream(true);
                  return item;
                } catch (error) {
                  finishError(error);
                  throw error;
                }
              };
            }
            if (property === "throw") {
              return (...throwArgs: any[]) => {
                try {
                  if (!target.throw) throw throwArgs[0];
                  const item = withNeatlogsSpan(span, () =>
                    (
                      target.throw as (
                        ...args: any[]
                      ) => IteratorResult<unknown>
                    )(...throwArgs),
                  );
                  if (item.done) finishStream(streamInterrupted);
                  else recordChunk(item.value);
                  return item;
                } catch (error) {
                  finishError(error);
                  throw error;
                }
              };
            }
            const value = Reflect.get(target, property, target);
            return typeof value === "function" ? value.bind(target) : value;
          },
        });
        return proxy;
      };

      const wrapReader = (reader: any): any =>
        new Proxy(reader, {
          get(target, property) {
            if (property === "read") {
              return async (...readArgs: any[]) => {
                try {
                  const item = await withNeatlogsSpan(span, () =>
                    target.read(...readArgs),
                  );
                  if (item.done) finishStream();
                  else recordChunk(item.value);
                  return item;
                } catch (error) {
                  finishError(error);
                  throw error;
                }
              };
            }
            if (property === "cancel") {
              return async (...cancelArgs: any[]) => {
                try {
                  const result = await target.cancel(...cancelArgs);
                  finishStream(true);
                  return result;
                } catch (error) {
                  finishError(error);
                  throw error;
                }
              };
            }
            if (property === "releaseLock") {
              return (...releaseArgs: any[]) => {
                try {
                  return target.releaseLock(...releaseArgs);
                } finally {
                  finishStream(true);
                }
              };
            }
            const value = Reflect.get(target, property, target);
            return typeof value === "function" ? value.bind(target) : value;
          },
        });

      const wrapReadableStream = (stream: any): any =>
        new Proxy(stream, {
          get(target, property) {
            if (property === "getReader") {
              return (...readerArgs: any[]) =>
                wrapReader(target.getReader(...readerArgs));
            }
            if (property === Symbol.asyncIterator) {
              const iteratorFactory = target[Symbol.asyncIterator];
              if (typeof iteratorFactory === "function") {
                return () => wrapAsyncIterator(iteratorFactory.call(target));
              }
            }
            if (property === "cancel") {
              return async (...cancelArgs: any[]) => {
                try {
                  const result = await target.cancel(...cancelArgs);
                  finishStream(true);
                  return result;
                } catch (error) {
                  finishError(error);
                  throw error;
                }
              };
            }
            if (property === "pipeTo") {
              return async (...pipeArgs: any[]) => {
                try {
                  const result = await target.pipeTo(...pipeArgs);
                  finishStream();
                  return result;
                } catch (error) {
                  finishError(error);
                  throw error;
                }
              };
            }
            const value = Reflect.get(target, property, target);
            return typeof value === "function" ? value.bind(target) : value;
          },
        });

      try {
        // Set common attributes
        setCommonSpanAttrs(span, opts);

        // Session/end-user identity (root span only; skipped on a non-root child).
        const spanIsRoot = isRoot && !autoRoot.root;
        applySessionAttributes(span, opts.sessionId, spanIsRoot, {
          parentSessionId: opts.parentSessionId,
          sessionFeatureName: opts.sessionFeatureName,
          sessionEntryPoint: opts.sessionEntryPoint,
        });
        applyEndUserAttributes(
          span,
          opts.endUserId,
          opts.endUserMetadata,
          spanIsRoot,
        );

        // Capture input
        if (captureInput && args.length > 0) {
          try {
            const inputValue =
              args.length === 1
                ? serializeObj(args[0])
                : args.map(serializeObj);
            span.setAttribute(
              "input.value",
              safeJsonDumps(
                captureMedia(
                  span,
                  mediaPrefix("input"),
                  inputValue,
                  "input",
                ),
              ),
            );
          } catch (err) {
            logger.debug(`Failed to capture input: ${err}`);
          }
        }

        // Execute the function
        const replay = opts.replay && (opts.kind === 'TOOL' || opts.kind === 'MCP_TOOL')
          ? lookupReplay('tool', opts.toolName ?? spanName, args.length === 1 ? args[0] : args)
          : null;
        const returnsPromise = opts.replay && typeof opts.replay === 'object'
          ? opts.replay.result === 'promise'
          : fn.constructor.name === 'AsyncFunction';
        const result = replay?.outcome === 'HIT'
          ? returnsPromise ? Promise.resolve(replay.entry.output) : replay.entry.output
          : fn(...args);

        const handleResult = (resolved: any): any => {
          // A promise may resolve to a stream. Classify the resolved value before
          // ending the span so async factories remain open through consumption.
          if (isReadableStreamLike(resolved))
            return wrapReadableStream(resolved);
          if (isAsyncIterable(resolved)) return wrapAsyncIterable(resolved);
          if (isSyncIterator(resolved)) return wrapSyncIterator(resolved);
          finishSuccess(resolved);
          return resolved;
        };

        // Handle any Promise/A+ thenable, not only native Promise instances.
        if (isPromiseLike(result)) {
          return Promise.resolve(result)
            .then(handleResult)
            .catch((error: any) => {
              finishError(error);
              throw error;
            });
        }

        return handleResult(result);
      } catch (error: any) {
        finishError(error);
        throw error;
      }
    }, autoRoot.ctx, autoRoot.root);
  };

  // Preserve function name for debugging
  Object.defineProperty(wrapped, "name", {
    value: spanName,
    configurable: true,
  });

  return wrapped as any;
}

/** Extract named arguments from function call (best-effort). */
function _extractBoundInputs(fn: Function, args: any[]): Record<string, any> {
  const result: Record<string, any> = {};
  // Parse function parameter names from toString()
  const fnStr = fn.toString();
  const match = fnStr.match(/\(([^)]*)\)/);
  if (match) {
    const paramNames = match[1]
      .split(",")
      .map((p) =>
        p
          .trim()
          .replace(/\s*[:=].*$/, "")
          .replace(/^\.\.\./, ""),
      )
      .filter(Boolean);
    for (let i = 0; i < Math.min(paramNames.length, args.length); i++) {
      result[paramNames[i]] = args[i];
    }
  }
  return result;
}
