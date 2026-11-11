/**
 * OpenAI Responses API span finalization, shared by the OpenAI and Azure
 * OpenAI wrappers (same wire format, different span name / provider).
 */

import { SpanStatusCode, type Span } from "@opentelemetry/api";
import {
  DEFAULT_MAX_SEMANTIC_STREAM_EVENTS,
  DEFAULT_MAX_STREAM_CAPTURE_BYTES,
  DEFAULT_MAX_STREAM_CAPTURE_ITEMS,
  utf8ByteLength,
} from "../constants.js";
import { captureMedia } from "./media.js";
import { instrumentSdkStreamIterator } from "./sdk-stream.js";
import {
  markCancelled,
  recordFailure,
  recordReportedFailure,
} from "./span-outcome.js";

/**
 * Keep a Responses API LLM span open until a streamed response is consumed.
 * `responses.create({ stream: true })` resolves to an async iterable immediately;
 * the assistant text and usage arrive later in `response.*` events.
 */
export function wrapResponsesAsyncIterableStream(stream: any, span: Span): any {
  const originalAsyncIterator = stream?.[Symbol.asyncIterator]?.bind(stream);

  if (!originalAsyncIterator) {
    // Be defensive about clients that ignore `stream: true` and return a normal
    // response object.
    finalizeResponsesResponse(span, stream);
    return stream;
  }

  const textParts: string[] = [];
  const toolCalls: any[] = [];
  const streamBudget = newStreamCaptureBudget();
  // The last response.completed / .failed / .incomplete payload.
  let terminalResponse: any;
  let errorEvent: any;
  let finalized = false;
  let eventCount = 0;
  let cancellationRequested = false;

  const setEventCounts = () => {
    span.setAttribute("neatlogs.stream.chunk_count", eventCount);
    if (eventCount > DEFAULT_MAX_SEMANTIC_STREAM_EVENTS) {
      span.setAttribute(
        "neatlogs.stream.events_dropped",
        eventCount - DEFAULT_MAX_SEMANTIC_STREAM_EVENTS,
      );
    }
    applyStreamDiagnostics(span, streamBudget);
  };

  const finalize = (interrupted = false) => {
    if (finalized) return;
    finalized = true;
    setEventCounts();
    finalizeResponsesResponse(
      span,
      terminalResponse,
      textParts.join(""),
      interrupted,
      toolCalls,
      errorEvent,
    );
  };

  const fail = (error: unknown) => {
    if (finalized) return;
    finalized = true;
    setEventCounts();
    applyResponsesResponseAttributes(
      span,
      terminalResponse,
      textParts.join(""),
      toolCalls,
    );
    recordFailure(span, error);
    span.end();
  };

  // The OpenAI SDK ends the iterator quietly when the request is aborted, so
  // an abort with no terminal event would otherwise look like a clean finish.
  const abortedBySdk = () =>
    !terminalResponse && !errorEvent && stream?.controller?.signal?.aborted === true;

  const addEvent = (event: any) => {
    const index = eventCount++;
    if (index < DEFAULT_MAX_SEMANTIC_STREAM_EVENTS) {
      const delta = typeof event?.delta === "string" ? event.delta : "";
      span.addEvent("neatlogs.stream.chunk", {
        "neatlogs.stream.chunk.index": index,
        "neatlogs.stream.chunk.summary": safeStringify({
          type: event?.type ?? "unknown",
          delta_bytes: utf8ByteLength(delta, DEFAULT_MAX_STREAM_CAPTURE_BYTES),
          has_response: event?.response != null,
        }),
      });
    }
    if (
      event?.type === "response.output_text.delta" &&
      typeof event.delta === "string"
    ) {
      retainStreamString(streamBudget, event.delta, (value) => {
        textParts.push(value);
      });
    } else if (
      event?.type === "response.output_item.done" &&
      event.item?.type === "function_call"
    ) {
      // Kept for a stream that breaks before its terminal event.
      retainStreamString(
        streamBudget,
        safeStringify(event.item.arguments ?? ""),
        () => {
          toolCalls.push(event.item);
        },
      );
    } else if (
      event?.type === "response.completed" ||
      event?.type === "response.failed" ||
      event?.type === "response.incomplete"
    ) {
      terminalResponse = event.response;
    } else if (event?.type === "error") {
      errorEvent = event;
    }
  };

  const wrapIterator = (iterator: AsyncIterator<any>): AsyncIterator<any> => ({
    async next(): Promise<IteratorResult<any>> {
      try {
        const result = await iterator.next();
        if (result.done) {
          finalize(cancellationRequested || abortedBySdk());
          return result;
        }
        addEvent(result.value);
        return result;
      } catch (err) {
        fail(err);
        throw err;
      }
    },
    async return(value?: any): Promise<IteratorResult<any>> {
      cancellationRequested = true;
      try {
        const result = await (iterator.return?.(value) ?? {
          done: true,
          value,
        });
        if (result.done) finalize(true);
        else addEvent(result.value);
        return result;
      } catch (err) {
        fail(err);
        throw err;
      }
    },
    async throw(err?: any): Promise<IteratorResult<any>> {
      fail(err);
      if (iterator.throw) return iterator.throw(err);
      throw err;
    },
  });

  if (instrumentSdkStreamIterator(stream, wrapIterator)) return stream;

  // A proxy preserves the OpenAI stream's public surface. Binding methods back
  // to the original object also avoids breaking SDK classes with private fields.
  return new Proxy(stream, {
    get(target, prop) {
      if (prop === Symbol.asyncIterator) {
        return () => wrapIterator(originalAsyncIterator());
      }

      const value = Reflect.get(target, prop, target);
      return typeof value === "function" ? value.bind(target) : value;
    },
  });
}

/**
 * Set the final status and end the span. A response the API reports as failed
 * (`status: "failed"`, `response.error`, or an in-stream `error` event) is an
 * error even though nothing was thrown; `incomplete` is not.
 */
export function finalizeResponsesResponse(
  span: Span,
  response: any,
  streamedText = "",
  interrupted = false,
  streamedToolCalls: any[] = [],
  errorEvent?: any,
): void {
  applyResponsesResponseAttributes(
    span,
    response,
    streamedText,
    streamedToolCalls,
  );
  const failure = reportedFailure(response, errorEvent);
  if (failure) {
    recordReportedFailure(span, failure.type, failure.message);
  } else if (interrupted) {
    markCancelled(span);
  } else {
    span.setStatus({ code: SpanStatusCode.OK });
  }
  span.end();
}

function reportedFailure(
  response: any,
  errorEvent: any,
): { type: string; message: string } | undefined {
  if (errorEvent) {
    return {
      type: String(errorEvent.code ?? "error"),
      message: String(errorEvent.message ?? "Responses stream error"),
    };
  }
  const error = response?.error;
  if (!error && response?.status !== "failed") return undefined;
  return {
    type: String(error?.code ?? "response.failed"),
    message: String(error?.message ?? "Response failed"),
  };
}

function applyResponsesResponseAttributes(
  span: Span,
  response: any,
  streamedText = "",
  streamedToolCalls: any[] = [],
): void {
  const capturedOutput = response?.output
    ? captureMedia(
        span,
        "neatlogs.llm.output_messages.0",
        response.output,
        "output",
      )
    : response?.output;
  const text =
    response?.output_text ||
    streamedText ||
    extractResponsesOutputText(response?.output);
  if (text) {
    span.setAttribute("neatlogs.llm.output_messages.0.role", "assistant");
    span.setAttribute("neatlogs.llm.output_messages.0.content", text);
    span.setAttribute("output.value", text);
  } else if (response?.output) {
    // Tool-call-only responses still have a meaningful structured output.
    span.setAttribute("output.value", safeStringify(capturedOutput));
  }
  setFunctionToolCalls(
    span,
    Array.isArray(response?.output) ? response.output : streamedToolCalls,
  );

  if (response?.model)
    span.setAttribute("neatlogs.llm.model_name", response.model);
  // `incomplete` only says the reply was cut short; the reason says why.
  const finishReason = response?.incomplete_details?.reason ?? response?.status;
  if (finishReason)
    span.setAttribute("neatlogs.llm.finish_reason", String(finishReason));

  const usage = response?.usage;
  if (usage) {
    if (usage.input_tokens != null)
      span.setAttribute("neatlogs.llm.token_count.prompt", usage.input_tokens);
    if (usage.output_tokens != null)
      span.setAttribute(
        "neatlogs.llm.token_count.completion",
        usage.output_tokens,
      );
    if (usage.total_tokens != null)
      span.setAttribute("neatlogs.llm.token_count.total", usage.total_tokens);
    if (usage.input_tokens_details?.cached_tokens != null) {
      span.setAttribute(
        "neatlogs.llm.token_count.cache_read",
        usage.input_tokens_details.cached_tokens,
      );
    }
    if (usage.output_tokens_details?.reasoning_tokens != null) {
      span.setAttribute(
        "neatlogs.llm.token_count.reasoning",
        usage.output_tokens_details.reasoning_tokens,
      );
    }
  }
}

/** Responses `function_call` output items, under the chat-completions tool-call keys. */
function setFunctionToolCalls(span: Span, items: any[]): void {
  let index = 0;
  for (const item of items) {
    if (item?.type !== "function_call") continue;
    if (index >= DEFAULT_MAX_STREAM_CAPTURE_ITEMS) return;
    const prefix = `neatlogs.llm.tool_calls.${index}`;
    span.setAttribute(`${prefix}.id`, String(item.call_id ?? item.id ?? ""));
    span.setAttribute(`${prefix}.type`, "function");
    span.setAttribute(`${prefix}.name`, String(item.name ?? ""));
    span.setAttribute(
      `${prefix}.arguments`,
      sanitizedJsonToolArguments(
        span,
        `${prefix}.arguments`,
        item.arguments ?? "{}",
      ),
    );
    index += 1;
  }
}

function extractResponsesOutputText(output: any): string {
  if (!Array.isArray(output)) return "";
  const parts: string[] = [];
  for (const item of output) {
    if (item?.type !== "message" || !Array.isArray(item.content)) continue;
    for (const content of item.content) {
      if (content?.type === "output_text" && typeof content.text === "string") {
        parts.push(content.text);
      }
    }
  }
  return parts.join("");
}

interface StreamCaptureBudget {
  capturedBytes: number;
  capturedItems: number;
  droppedBytes: number;
  droppedItems: number;
  incompleteReasons: Set<string>;
}

function newStreamCaptureBudget(): StreamCaptureBudget {
  return {
    capturedBytes: 0,
    capturedItems: 0,
    droppedBytes: 0,
    droppedItems: 0,
    incompleteReasons: new Set(),
  };
}

function retainStreamString(
  budget: StreamCaptureBudget,
  value: string,
  retain: (value: string) => void,
): void {
  const remaining = Math.max(0, DEFAULT_MAX_STREAM_CAPTURE_BYTES - budget.capturedBytes);
  const bytes = utf8ByteLength(value, remaining);
  if (budget.capturedItems >= DEFAULT_MAX_STREAM_CAPTURE_ITEMS) {
    budget.droppedItems += 1;
    budget.droppedBytes += bytes;
    budget.incompleteReasons.add("item_limit_exceeded");
    return;
  }
  if (budget.capturedBytes + bytes > DEFAULT_MAX_STREAM_CAPTURE_BYTES) {
    budget.droppedItems += 1;
    budget.droppedBytes += bytes;
    budget.incompleteReasons.add("byte_limit_exceeded");
    return;
  }
  retain(value);
  budget.capturedItems += 1;
  budget.capturedBytes += bytes;
}

function applyStreamDiagnostics(span: Span, budget: StreamCaptureBudget): void {
  span.setAttribute("neatlogs.stream.capture_bytes", budget.capturedBytes);
  span.setAttribute("neatlogs.stream.capture_items", budget.capturedItems);
  if (budget.incompleteReasons.size === 0) return;
  span.setAttribute("neatlogs.stream.incomplete", true);
  span.setAttribute(
    "neatlogs.stream.incomplete_reason",
    [...budget.incompleteReasons].sort().join(","),
  );
  if (budget.droppedBytes > 0) {
    span.setAttribute("neatlogs.stream.dropped_bytes", budget.droppedBytes);
    span.setAttribute("neatlogs.stream.dropped_bytes_is_lower_bound", true);
  }
  span.setAttribute("neatlogs.stream.dropped_items", budget.droppedItems);
  span.setAttribute("neatlogs.capture_fidelity", "truncated");
}

export function sanitizedJsonToolArguments(
  span: Span,
  prefix: string,
  value: unknown,
): string {
  if (typeof value !== "string") {
    return safeStringify(captureMedia(span, prefix, value, "output"));
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(value);
  } catch {
    span.setAttribute("neatlogs.stream.incomplete", true);
    span.setAttribute("neatlogs.stream.incomplete_reason", "invalid_tool_arguments");
    return "[incomplete: invalid tool arguments omitted]";
  }
  const original = safeStringify(parsed);
  const sanitized = safeStringify(captureMedia(span, prefix, parsed, "output"));
  return original === sanitized ? value : sanitized;
}

function safeStringify(value: unknown): string {
  try {
    return typeof value === "string" ? value : JSON.stringify(value);
  } catch {
    return "";
  }
}
