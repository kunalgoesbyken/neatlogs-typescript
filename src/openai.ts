/**
 * Neatlogs OpenAI wrapper — ES6 Proxy-based.
 *
 * Usage:
 *   import { wrapOpenAI, traceTool } from 'neatlogs/openai';
 *   import OpenAI from 'openai';
 *   const client = wrapOpenAI(new OpenAI());
 *
 * Intercepts:
 *   - client.chat.completions.create() — non-streaming and streaming
 *   - client.responses.create() — Responses API
 *
 * Also exports traceTool() to wrap user-defined tool functions with TOOL spans.
 */

import { SpanStatusCode, type Span } from "@opentelemetry/api";
import { observeRawResponse } from "./core/raw-response.js";
import { getProviderTracer } from "./core/auto-root.js";
import { ChoiceAccumulator } from "./core/choice-accumulator.js";
import { captureMedia } from "./core/media.js";
import {
  finalizeResponsesResponse,
  sanitizedJsonToolArguments,
  wrapResponsesAsyncIterableStream,
} from "./core/openai-responses.js";
import { recordFailure } from "./core/span-outcome.js";
import { cachedModelSpan, providerReplay } from './replay/provider.js';
import { prepareProviderCapture } from './replay/provider-capture.js';
import {
  getNeatlogsTracer,
  getNeatlogsParentContext,
  withNeatlogsSpan,
} from "./core/provider.js";

const TRACER_NAME = "neatlogs.openai";

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export function wrapOpenAI<T extends object>(client: T): T {
  return wrapNamespace(client, []) as T;
}

/**
 * Wrap a tool/function implementation to emit TOOL spans when executed.
 *
 * Usage:
 *   const getWeather = traceTool('get_weather', async (args: { city: string }) => {
 *     return `Weather in ${args.city}: sunny`;
 *   });
 */
export function traceTool<TArgs = any, TResult = any>(
  name: string,
  fn: (args: TArgs) => TResult | Promise<TResult>,
): (args: TArgs) => Promise<TResult> {
  return async function tracedTool(args: TArgs): Promise<TResult> {
    const tracer = getNeatlogsTracer(TRACER_NAME);
    const span = tracer.startSpan(
      `tool.${name}`,
      {
        attributes: {
          "neatlogs.span.kind": "TOOL",
          "neatlogs.tool.name": name,
        },
      },
      getNeatlogsParentContext(),
    );
    span.setAttribute(
      "input.value",
      safeStringify(captureMedia(span, "neatlogs.tool.input", args, "input")),
    );
    return withNeatlogsSpan(span, async () => {
      try {
        const result = await fn(args);
        span.setAttribute(
          "output.value",
          safeStringify(captureMedia(span, "neatlogs.tool.output", result, "output")),
        );
        span.setStatus({ code: SpanStatusCode.OK });
        return result;
      } catch (err) {
        recordError(span, err);
        throw err;
      } finally {
        span.end();
      }
    });
  };
}

// ---------------------------------------------------------------------------
// Proxy wrapping
// ---------------------------------------------------------------------------

function wrapNamespace(target: any, path: string[]): any {
  return new Proxy(target, {
    get(obj, prop, receiver) {
      const value = Reflect.get(obj, prop, receiver);
      if (typeof prop === "symbol" || String(prop).startsWith("_"))
        return value;

      const currentPath = [...path, String(prop)];
      const pathStr = currentPath.join(".");

      if (
        pathStr === "chat.completions.create" &&
        typeof value === "function"
      ) {
        return tracedChatCompletionsCreate(value.bind(obj));
      }
      if (pathStr === "responses.create" && typeof value === "function") {
        return tracedResponsesCreate(value.bind(obj));
      }

      if (
        value &&
        typeof value === "object" &&
        !Array.isArray(value) &&
        isNamespace(currentPath)
      ) {
        return wrapNamespace(value, currentPath);
      }

      return value;
    },
  });
}

function isNamespace(path: string[]): boolean {
  if (path.length > 3) return false;
  const key = path[path.length - 1];
  return ["chat", "completions", "responses", "beta"].includes(key);
}

// ---------------------------------------------------------------------------
// chat.completions.create — non-streaming + streaming
// ---------------------------------------------------------------------------

function tracedChatCompletionsCreate(original: (...args: any[]) => any) {
  return function (opts: any, ...rest: any[]): any {
    const tracer = getProviderTracer(TRACER_NAME);
    const model = opts?.model ?? "";
    const messages: any[] = opts?.messages ?? [];
    const isStream = opts?.stream === true;

    const span = tracer.startSpan(
      "openai.chat.completions.create",
      {
        attributes: {
          "neatlogs.span.kind": "LLM",
          "neatlogs.llm.provider": "openai",
          "neatlogs.llm.system": "openai",
          "neatlogs.llm.model_name": model,
          "neatlogs.llm.is_streaming": isStream,
        },
      },
      getNeatlogsParentContext(),
    );

    // Input messages
    for (let i = 0; i < messages.length; i++) {
      const msg = messages[i];
      span.setAttribute(
        `neatlogs.llm.input_messages.${i}.role`,
        msg.role ?? "",
      );
      if (typeof msg.content === "string") {
        span.setAttribute(
          `neatlogs.llm.input_messages.${i}.content`,
          msg.content,
        );
      } else if (msg.content) {
        const capturedContent = captureMedia(
          span,
          `neatlogs.llm.input_messages.${i}`,
          msg.content,
          "input",
        );
        span.setAttribute(
          `neatlogs.llm.input_messages.${i}.content`,
          safeStringify(capturedContent),
        );
      }
      if (msg.tool_call_id) {
        span.setAttribute(
          `neatlogs.llm.input_messages.${i}.tool_call_id`,
          msg.tool_call_id,
        );
      }
    }

    // Tools
    if (opts?.tools) {
      for (let i = 0; i < opts.tools.length; i++) {
        const tool = opts.tools[i] ?? {};
        const fn = tool.function ?? {};
        span.setAttribute(
          `neatlogs.llm.tools.${i}.type`,
          tool.type ?? "function",
        );
        span.setAttribute(
          `neatlogs.llm.tools.${i}.definition`,
          safeStringify(tool),
        );
        span.setAttribute(`neatlogs.llm.tools.${i}.name`, fn.name ?? "");
        if (fn.description)
          span.setAttribute(
            `neatlogs.llm.tools.${i}.description`,
            fn.description,
          );
        if (fn.parameters)
          span.setAttribute(
            `neatlogs.llm.tools.${i}.input_schema`,
            safeStringify(fn.parameters),
          );
      }
    }

    // Invocation parameters
    setInvocationParams(span, opts);

    // Force stream_options.include_usage for streaming
    if (isStream) {
      opts = { ...opts };
      const streamOpts = opts.stream_options ?? {};
      if (!streamOpts.include_usage) {
        opts.stream_options = { ...streamOpts, include_usage: true };
      }
    }

    const captureResponse = prepareProviderCapture(span, 'openai.chat.completions.create', opts, !isStream && rest.length === 0);
    let result: any;
    let replayed = false;
    try {
      result = withNeatlogsSpan(span, () => {
        const saved = providerReplay('openai.chat.completions.create', opts, !isStream && rest.length === 0);
        replayed = saved !== undefined;
        return replayed ? saved : original(opts, ...rest);
      });
    } catch (err) {
      recordError(span, err);
      throw err;
    }

    return preserveResponseHelpers(result, span,
      (response: any) => {
        if (isStream) {
          return wrapAsyncIterableStream(response, span);
        }
        if (!replayed) captureResponse(response);
        finalizeChatResponse(replayed ? cachedModelSpan(span) : span, response);
        return response;
      },
      isStream,
    );
  };
}

// ---------------------------------------------------------------------------
// responses.create
// ---------------------------------------------------------------------------


// Preserve the SDK's response helpers while sharing one traced parsed result.
function preserveResponseHelpers(result: any, span: Span, finalize: (value: any) => any, isStream = false): any {
  const target = result && typeof result.then === "function" ? result : Promise.resolve(result);
  let traced: Promise<any> | undefined;
  const consume = () => traced ??= Promise.resolve(target).then(finalize, (err: any) => {
    recordError(span, err);
    throw err;
  });
  return new Proxy(target, {
    get(promise, prop) {
      if (prop === "then" || prop === "catch" || prop === "finally") {
        return (...args: any[]) => (consume() as any)[prop](...args);
      }
      if (prop === "withResponse" && typeof promise.withResponse === "function") {
        return async () => {
          const [data, response] = await Promise.all([consume(), promise.asResponse()]);
          return { data, response, request_id: response.headers.get("x-request-id") };
        };
      }
      if (prop === "asResponse" && typeof promise.asResponse === "function") {
        return async () => {
          try {
            const response = await promise.asResponse();
            if (!traced && span.isRecording()) {
              if (response.ok) {
                observeRawResponse(response, span, {
                  isStream,
                  isTraced: () => traced !== undefined,
                  finalize,
                });
              } else {
                span.setStatus({ code: SpanStatusCode.ERROR });
                span.end();
              }
            }
            return response;
          } catch (err) {
            recordError(span, err);
            throw err;
          }
        };
      }
      const value = Reflect.get(promise, prop, promise);
      return typeof value === "function" ? value.bind(promise) : value;
    },
  });
}

function tracedResponsesCreate(original: (...args: any[]) => any) {
  return function (opts: any, ...rest: any[]): any {
    const tracer = getProviderTracer(TRACER_NAME);
    const model = opts?.model ?? "";
    const isStream = opts?.stream === true;

    const span = tracer.startSpan(
      "openai.responses.create",
      {
        attributes: {
          "neatlogs.span.kind": "LLM",
          "neatlogs.llm.provider": "openai",
          "neatlogs.llm.system": "openai",
          "neatlogs.llm.model_name": model,
          "neatlogs.llm.is_streaming": isStream,
          "input.value": "",
        },
      },
      getNeatlogsParentContext(),
    );

    const capturedInput = captureMedia(
      span,
      "neatlogs.llm.input_messages.0",
      opts?.input,
      "input",
    );
    span.setAttribute("input.value", safeStringify(capturedInput ?? ""));
    setInvocationParams(span, opts);

    const captureResponse = prepareProviderCapture(span, 'openai.responses.create', opts, !isStream && rest.length === 0);
    let result: any;
    let replayed = false;
    try {
      result = withNeatlogsSpan(span, () => {
        const saved = providerReplay('openai.responses.create', opts, !isStream && rest.length === 0);
        replayed = saved !== undefined;
        return replayed ? saved : original(opts, ...rest);
      });
    } catch (err) {
      recordError(span, err);
      throw err;
    }

    return preserveResponseHelpers(result, span,
      (response: any) => {
        if (isStream) {
          return wrapResponsesAsyncIterableStream(response, span);
        }
        if (!replayed) captureResponse(response);
        finalizeResponsesResponse(replayed ? cachedModelSpan(span) : span, response);
        return response;
      },
      isStream,
    );
  };
}

// ---------------------------------------------------------------------------
// Streaming support
// ---------------------------------------------------------------------------

function wrapAsyncIterableStream(stream: any, span: Span): any {
  const accumulator = new ChoiceAccumulator();
  const safeSpan = mediaSafeAccumulatorSpan(span);
  let cancellationRequested = false;
  const originalAsyncIterator = stream[Symbol.asyncIterator]?.bind(stream);

  if (!originalAsyncIterator) {
    accumulator.addResponse(stream, span);
    accumulator.finish(safeSpan);
    return stream;
  }

  return new Proxy(stream, {
    get(target, property) {
      if (property === Symbol.asyncIterator) {
        return () => {
          const iterator = originalAsyncIterator();
          return {
            async next(): Promise<IteratorResult<any>> {
              try {
                const result = await iterator.next();
                if (result.done) {
                  // The SDK ends the iterator quietly on an abort; a chunk with
                  // a finish_reason means the reply did complete first.
                  accumulator.finish(
                    safeSpan,
                    cancellationRequested ||
                      (stream.controller?.signal?.aborted === true &&
                        !accumulator.hasFinishReason()),
                  );
                  return result;
                }
                accumulator.addChunk(span, result.value);
                return result;
              } catch (err) {
                accumulator.fail(safeSpan, err);
                throw err;
              }
            },
            async return(value?: any): Promise<IteratorResult<any>> {
              cancellationRequested = true;
              try {
                const result = await (iterator.return?.(value) ?? {
                  done: true,
                  value: undefined,
                });
                if (result.done) accumulator.finish(safeSpan, true);
                else accumulator.addChunk(span, result.value);
                return result;
              } catch (err) {
                accumulator.fail(safeSpan, err);
                throw err;
              }
            },
            async throw(err?: any): Promise<IteratorResult<any>> {
              accumulator.fail(safeSpan, err);
              if (iterator.throw) return iterator.throw(err);
              throw err;
            },
          };
        };
      }
      const value = Reflect.get(target, property, target);
      return typeof value === "function" ? value.bind(target) : value;
    },
  });
}

// ---------------------------------------------------------------------------
// Non-streaming response finalization
// ---------------------------------------------------------------------------

function finalizeChatResponse(span: Span, response: any): void {
  const accumulator = new ChoiceAccumulator();
  accumulator.addResponse(response, span);
  accumulator.finish(mediaSafeAccumulatorSpan(span));
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function mediaSafeAccumulatorSpan(span: Span): Span {
  return new Proxy(span, {
    get(target, property) {
      if (property === "setAttribute") {
        return (name: string, value: unknown) =>
          target.setAttribute(
            name,
            /^neatlogs\.llm\.tool_calls\.\d+\.arguments$/.test(name)
              ? sanitizedJsonToolArguments(target, name, value)
              : (value as any),
          );
      }
      const value = Reflect.get(target, property, target);
      return typeof value === "function" ? value.bind(target) : value;
    },
  });
}

function setInvocationParams(span: Span, opts: any): void {
  if (opts?.temperature != null)
    span.setAttribute("neatlogs.llm.temperature", opts.temperature);
  if (opts?.top_p != null) span.setAttribute("neatlogs.llm.top_p", opts.top_p);
  if (opts?.max_tokens != null)
    span.setAttribute("neatlogs.llm.max_tokens", opts.max_tokens);
  if (opts?.frequency_penalty != null)
    span.setAttribute("neatlogs.llm.frequency_penalty", opts.frequency_penalty);
  if (opts?.presence_penalty != null)
    span.setAttribute("neatlogs.llm.presence_penalty", opts.presence_penalty);
  if (opts?.stop)
    span.setAttribute("neatlogs.llm.stop_sequences", safeStringify(opts.stop));
}

function safeStringify(value: unknown): string {
  try {
    return typeof value === "string" ? value : JSON.stringify(value);
  } catch {
    return "";
  }
}

function recordError(span: Span, err: unknown): void {
  recordFailure(span, err);
  span.end();
}
