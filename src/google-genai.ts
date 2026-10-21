/**
 * Neatlogs Google Gemini wrapper — ES6 Proxy-based.
 *
 * Gemini (Google AI Studio) is accessed through the `@google/genai` SDK
 * (`new GoogleGenAI({ apiKey })`). This wrapper traces those clients with
 * provider `google` so Gemini traffic is distinguished from Vertex AI traffic
 * (see `wrapVertexAI`, which is the same SDK in `vertexai: true` mode).
 *
 * Mirrors Python's `neatlogs.wrap(genai.Client())` with explicit, private-provider
 * instrumentation.
 *
 * Usage:
 *   import { wrapGoogleGenAI } from 'neatlogs/google-genai';
 *   import { GoogleGenAI } from '@google/genai';
 *   const client = wrapGoogleGenAI(new GoogleGenAI({ apiKey: process.env.GOOGLE_API_KEY }));
 *   const res = await client.models.generateContent({ model: 'gemini-2.5-flash', contents: 'Hello' });
 *
 * Intercepts:
 *   - client.models.generateContent()        — non-streaming
 *   - client.models.generateContentStream()  — streaming (async iterable)
 *   - client.models.embedContent()           — embeddings
 *   - client.models.countTokens()            — token counting
 *   - chat.sendMessage() / sendMessageStream() — chat sessions
 */

import { SpanStatusCode, type Span } from '@opentelemetry/api';
import {
  DEFAULT_MAX_STREAM_CAPTURE_BYTES,
  DEFAULT_MAX_STREAM_CAPTURE_ITEMS,
  utf8ByteLength,
} from './constants.js';
import { getProviderTracer } from './core/auto-root.js';
import {
  markCancelled,
  recordFailure,
  recordReportedFailure,
} from './core/span-outcome.js';
import { captureMedia, captureMediaWithIndex } from './core/media.js';
import { cachedModelSpan, providerReplay, restoreGoogleResponse } from './replay/provider.js';
import { prepareProviderCapture } from './replay/provider-capture.js';
import {
  getNeatlogsTracer,
  getNeatlogsParentContext,
  withNeatlogsSpan,
} from './core/provider.js';

const TRACER_NAME = 'neatlogs.google_genai';
const PROVIDER = 'google';
const SYSTEM = 'google_genai';

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export function wrapGoogleGenAI<T extends object>(client: T): T {
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
          'neatlogs.span.kind': 'TOOL',
          'neatlogs.tool.name': name,
        },
      },
      getNeatlogsParentContext(),
    );
    span.setAttribute(
      'input.value',
      safeStringify(captureMedia(span, 'neatlogs.tool.input', args, 'input')),
    );
    return withNeatlogsSpan(span, async () => {
      try {
        const result = await fn(args);
        span.setAttribute(
          'output.value',
          safeStringify(captureMedia(span, 'neatlogs.tool.output', result, 'output')),
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
      if (typeof prop === 'symbol' || String(prop).startsWith('_')) return value;

      const currentPath = [...path, String(prop)];
      const pathStr = currentPath.join('.');

      if (pathStr === 'models.generateContent' && typeof value === 'function') {
        return tracedGenerateContent(value.bind(obj), false);
      }
      if (pathStr === 'models.generateContentStream' && typeof value === 'function') {
        return tracedGenerateContent(value.bind(obj), true);
      }
      if (pathStr === 'models.embedContent' && typeof value === 'function') {
        return tracedEmbedContent(value.bind(obj));
      }
      if (pathStr === 'models.countTokens' && typeof value === 'function') {
        return tracedCountTokens(value.bind(obj));
      }

      if (value && typeof value === 'object' && !Array.isArray(value) && isNamespace(currentPath)) {
        return wrapNamespace(value, currentPath);
      }

      return value;
    },
  });
}

function isNamespace(path: string[]): boolean {
  if (path.length > 2) return false;
  return ['models', 'chats'].includes(path[path.length - 1]);
}

/**
 * Wrap a chat session returned by `client.chats.create(...)`. The `@google/genai`
 * Chat object exposes `sendMessage` / `sendMessageStream`; wrap them so each turn
 * emits an LLM span. Returns the same chat instance.
 */
export function wrapGoogleGenAIChat<T extends object>(chat: T): T {
  const c = chat as any;
  if (!c || c._neatlogsGoogleGenAIPatched) return chat;

  if (typeof c.sendMessage === 'function') {
    const orig = c.sendMessage.bind(c);
    c.sendMessage = (params: any, ...rest: any[]) => tracedChatSend(orig, c, params, rest, false);
  }
  if (typeof c.sendMessageStream === 'function') {
    const orig = c.sendMessageStream.bind(c);
    c.sendMessageStream = (params: any, ...rest: any[]) => tracedChatSend(orig, c, params, rest, true);
  }

  try {
    Object.defineProperty(c, '_neatlogsGoogleGenAIPatched', { value: true, enumerable: false, configurable: true });
  } catch {
    c._neatlogsGoogleGenAIPatched = true;
  }
  return chat;
}

function tracedChatSend(
  original: (...a: any[]) => any,
  chat: any,
  params: any,
  rest: any[],
  isStream: boolean,
): any {
  const tracer = getProviderTracer(TRACER_NAME);
  const model = chat?.model ?? chat?.modelVersion ?? '';
  const span = tracer.startSpan('google_genai.chat.send_message', {
    attributes: {
      'neatlogs.span.kind': 'LLM',
      'neatlogs.llm.provider': PROVIDER,
      'neatlogs.llm.system': SYSTEM,
      'neatlogs.llm.model_name': model,
      'neatlogs.llm.is_streaming': isStream,
    },
  }, getNeatlogsParentContext());

  // The chat sendMessage param shape is { message } (string or Part[]).
  const message = params?.message ?? params;
  span.setAttribute('neatlogs.llm.input_messages.0.role', 'user');
  const capturedMessage = captureMedia(
    span,
    'neatlogs.llm.input_messages.0',
    message,
    'input',
  );
  span.setAttribute(
    'neatlogs.llm.input_messages.0.content',
    (typeof capturedMessage === 'string' ? capturedMessage : safeStringify(capturedMessage)),
  );

  const result = withNeatlogsSpan(span, () => {
    // The SDK owns mutable chat history. A saved JSON reply cannot update it.
    providerReplay('google_genai.models.generate_content', params, false);
    return original(params, ...rest);
  });

  return Promise.resolve(result).then(
    (response: any) => {
      if (isStream) return wrapStream(response, span);
      finalizeResponse(span, response);
      return response;
    },
    (err: any) => {
      recordError(span, err);
      throw err;
    },
  );
}

function tracedEmbedContent(original: (...a: any[]) => any) {
  return function (opts: any, ...rest: any[]): any {
    const tracer = getProviderTracer(TRACER_NAME);
    const span = tracer.startSpan('google_genai.models.embed_content', {
      attributes: {
        'neatlogs.span.kind': 'EMBEDDING',
        'neatlogs.llm.provider': PROVIDER,
        'neatlogs.embedding.model_name': opts?.model ?? '',
        'neatlogs.embedding.text': safeStringify(opts?.contents ?? ''),
      },
    }, getNeatlogsParentContext());

    const result = withNeatlogsSpan(span, () => original(opts, ...rest));

    return Promise.resolve(result).then(
      (response: any) => {
        const embeddings = response?.embeddings;
        if (Array.isArray(embeddings)) {
          span.setAttribute('neatlogs.embedding.count', embeddings.length);
          const vals = embeddings[0]?.values;
          if (Array.isArray(vals)) span.setAttribute('neatlogs.embedding.dimensions', vals.length);
        }
        span.setStatus({ code: SpanStatusCode.OK });
        span.end();
        return response;
      },
      (err: any) => {
        recordError(span, err);
        throw err;
      },
    );
  };
}

function tracedCountTokens(original: (...a: any[]) => any) {
  return function (opts: any, ...rest: any[]): any {
    const tracer = getProviderTracer(TRACER_NAME);
    const span = tracer.startSpan('google_genai.models.count_tokens', {
      attributes: {
        'neatlogs.span.kind': 'LLM',
        'neatlogs.llm.provider': PROVIDER,
        'neatlogs.llm.task': 'count_tokens',
        'neatlogs.llm.model_name': opts?.model ?? '',
      },
    }, getNeatlogsParentContext());

    const result = withNeatlogsSpan(span, () => original(opts, ...rest));

    return Promise.resolve(result).then(
      (response: any) => {
        if (response?.totalTokens != null) span.setAttribute('neatlogs.llm.token_count.prompt', response.totalTokens);
        span.setStatus({ code: SpanStatusCode.OK });
        span.end();
        return response;
      },
      (err: any) => {
        recordError(span, err);
        throw err;
      },
    );
  };
}

// ---------------------------------------------------------------------------
// generateContent / generateContentStream
// ---------------------------------------------------------------------------

function tracedGenerateContent(original: (...args: any[]) => any, isStream: boolean) {
  return function (opts: any, ...rest: any[]): any {
    const tracer = getProviderTracer(TRACER_NAME);
    const model = opts?.model ?? '';

    const span = tracer.startSpan('google_genai.models.generate_content', {
      attributes: {
        'neatlogs.span.kind': 'LLM',
        'neatlogs.llm.provider': PROVIDER,
        'neatlogs.llm.system': SYSTEM,
        'neatlogs.llm.model_name': model,
        'neatlogs.llm.is_streaming': isStream,
      },
    }, getNeatlogsParentContext());

    setInputAttributes(span, opts);

    const captureResponse = prepareProviderCapture(span, 'google_genai.models.generate_content', opts, !isStream && rest.length === 0);
    let replayed = false;
    let result: unknown;
    try {
      result = withNeatlogsSpan(span, () => {
        const saved = providerReplay('google_genai.models.generate_content', opts, !isStream && rest.length === 0);
        replayed = saved !== undefined;
        return replayed ? restoreGoogleResponse(saved) : original(opts, ...rest);
      });
    } catch (error) {
      recordError(span, error);
      throw error;
    }

    return Promise.resolve(result).then(
      (response: any) => {
        if (isStream) {
          return wrapStream(response, span);
        }
        if (!replayed) captureResponse(response);
        finalizeResponse(replayed ? cachedModelSpan(span) : span, response);
        return response;
      },
      (err: any) => {
        recordError(span, err);
        throw err;
      },
    );
  };
}

function setInputAttributes(span: Span, opts: any): void {
  let idx = 0;

  const config = opts?.config;
  const systemInstruction = config?.systemInstruction ?? config?.system_instruction;
  if (systemInstruction) {
    const captured = captureMedia(
      span,
      `neatlogs.llm.input_messages.${idx}`,
      systemInstruction,
      'input',
    );
    span.setAttribute(`neatlogs.llm.input_messages.${idx}.role`, 'system');
    span.setAttribute(
      `neatlogs.llm.input_messages.${idx}.content`,
      typeof captured === 'string' ? captured : safeStringify(captured),
    );
    idx++;
  }

  const contents = opts?.contents;
  if (typeof contents === 'string') {
    span.setAttribute(`neatlogs.llm.input_messages.${idx}.role`, 'user');
    span.setAttribute(`neatlogs.llm.input_messages.${idx}.content`, contents);
  } else if (Array.isArray(contents) || (contents && typeof contents === 'object' && Array.isArray(contents.parts))) {
    for (const item of Array.isArray(contents) ? contents : [contents]) {
      if (typeof item === 'string') {
        span.setAttribute(`neatlogs.llm.input_messages.${idx}.role`, 'user');
        span.setAttribute(`neatlogs.llm.input_messages.${idx}.content`, item);
        idx++;
      } else if (item && typeof item === 'object') {
        const role = item.role ?? 'user';
        span.setAttribute(`neatlogs.llm.input_messages.${idx}.role`, role);
        const parts = item.parts ?? [];
        const capturedParts = captureMedia(
          span,
          `neatlogs.llm.input_messages.${idx}`,
          parts,
          'input',
        );
        const textParts: string[] = [];
        for (const part of parts) {
          if (typeof part === 'string') textParts.push(part);
          else if (part?.text) textParts.push(part.text);
        }
        span.setAttribute(
          `neatlogs.llm.input_messages.${idx}.content`,
          textParts.length ? textParts.join('\n') : safeStringify(capturedParts),
        );
        idx++;
      }
    }
  } else if (contents) {
    span.setAttribute(`neatlogs.llm.input_messages.${idx}.role`, 'user');
    const captured = captureMedia(
      span,
      `neatlogs.llm.input_messages.${idx}`,
      contents,
      'input',
    );
    span.setAttribute(`neatlogs.llm.input_messages.${idx}.content`, safeStringify(captured));
  }

  // Tools (function declarations)
  const tools = config?.tools;
  if (Array.isArray(tools)) {
    let t = 0;
    for (const tool of tools) {
      const decls = tool?.functionDeclarations ?? tool?.function_declarations ?? [];
      for (const fn of decls) {
        span.setAttribute(`neatlogs.llm.tools.${t}.name`, fn?.name ?? '');
        if (fn?.description) span.setAttribute(`neatlogs.llm.tools.${t}.description`, fn.description);
        if (fn?.parameters) span.setAttribute(`neatlogs.llm.tools.${t}.input_schema`, safeStringify(fn.parameters));
        t++;
      }
    }
  }

  // Invocation params
  if (config) {
    if (config.temperature != null) span.setAttribute('neatlogs.llm.temperature', config.temperature);
    if (config.topP != null) span.setAttribute('neatlogs.llm.top_p', config.topP);
    if (config.topK != null) span.setAttribute('neatlogs.llm.top_k', config.topK);
    const maxTokens = config.maxOutputTokens ?? config.max_output_tokens;
    if (maxTokens != null) span.setAttribute('neatlogs.llm.max_tokens', maxTokens);

    // Blob the backend reads for metadata.model_settings (the UI source).
    const params: Record<string, any> = {};
    if (config.temperature != null) params.temperature = config.temperature;
    if (config.topP != null) params.top_p = config.topP;
    if (config.topK != null) params.top_k = config.topK;
    if (maxTokens != null) params.max_tokens = maxTokens;
    if (config.frequencyPenalty != null) params.frequency_penalty = config.frequencyPenalty;
    if (config.presencePenalty != null) params.presence_penalty = config.presencePenalty;
    if (Object.keys(params).length > 0) {
      span.setAttribute('neatlogs.llm.invocation_parameters', JSON.stringify(params));
    }
  }
}

// ---------------------------------------------------------------------------
// Streaming support — @google/genai stream is an async iterable of chunks
// ---------------------------------------------------------------------------

function wrapStream(stream: any, span: Span): any {
  const accumulated: StreamAccumulator = {
    candidates: new Map(),
    usage: null,
    mediaCount: 0,
    capturedBytes: 0,
    capturedItems: 0,
    droppedBytes: 0,
    droppedItems: 0,
    incompleteReasons: new Set(),
    promptFeedback: undefined,
    finalized: false,
  };
  let cancellationRequested = false;
  const originalAsyncIterator = stream?.[Symbol.asyncIterator]?.bind(stream);

  if (!originalAsyncIterator) {
    finalizeStreamChunks(span, accumulated);
    return stream;
  }

  const wrapped = Object.create(Object.getPrototypeOf(stream));
  Object.assign(wrapped, stream);

  wrapped[Symbol.asyncIterator] = function () {
    const iterator = originalAsyncIterator();
    return {
      async next(): Promise<IteratorResult<any>> {
        try {
          const result = await iterator.next();
          if (result.done) {
            finalizeStreamChunks(span, accumulated, cancellationRequested);
            return result;
          }
          addStreamChunk(span, accumulated, result.value);
          return result;
        } catch (err) {
          failStreamChunks(span, accumulated, err);
          throw err;
        }
      },
      async return(value?: any): Promise<IteratorResult<any>> {
        cancellationRequested = true;
        markStreamIncomplete(accumulated, 'consumer_cancelled');
        span.setAttribute('neatlogs.stream.cancelled', true);
        try {
          const result = await (iterator.return?.(value) ?? { done: true, value: undefined });
          if (result.done) finalizeStreamChunks(span, accumulated, true);
          else addStreamChunk(span, accumulated, result.value);
          return result;
        } catch (err) {
          failStreamChunks(span, accumulated, err);
          throw err;
        }
      },
      async throw(err?: any): Promise<IteratorResult<any>> {
        failStreamChunks(span, accumulated, err);
        return iterator.throw?.(err) ?? { done: true, value: undefined };
      },
    };
  };

  return wrapped;
}

interface CandidateAccumulator {
  textParts: string[];
  thinkingParts: string[];
  toolCalls: Array<{ id: string; name: string; arguments: string }>;
  finishReason: string;
}

interface StreamAccumulator {
  candidates: Map<number, CandidateAccumulator>;
  usage: any;
  mediaCount: number;
  capturedBytes: number;
  capturedItems: number;
  droppedBytes: number;
  droppedItems: number;
  incompleteReasons: Set<string>;
  promptFeedback: any;
  finalized: boolean;
}

function candidateBucket(accumulated: StreamAccumulator, index: number): CandidateAccumulator {
  let bucket = accumulated.candidates.get(index);
  if (!bucket) {
    bucket = { textParts: [], thinkingParts: [], toolCalls: [], finishReason: '' };
    accumulated.candidates.set(index, bucket);
  }
  return bucket;
}

// Streamed text goes through media capture first so inline data URLs never reach attributes.
function safeStreamText(span: Span, accumulated: StreamAccumulator, candidateIndex: number, text: unknown): string {
  const captured = captureMediaWithIndex(
    span,
    `neatlogs.llm.output_messages.${candidateIndex}`,
    String(text),
    'output',
    accumulated.mediaCount,
  );
  accumulated.mediaCount += captured.count;
  if (typeof captured.value === 'string') return captured.value;
  markStreamIncomplete(accumulated, 'media_capture_incomplete');
  return '';
}

function addStreamChunk(span: Span, accumulated: StreamAccumulator, chunk: any): void {
  const chunkCandidates = chunk?.candidates ?? [];
  for (let position = 0; position < chunkCandidates.length; position++) {
    const candidate = chunkCandidates[position];
    const candidateIndex = typeof candidate?.index === 'number' ? candidate.index : position;
    const bucket = candidateBucket(accumulated, candidateIndex);
    for (const part of candidate?.content?.parts ?? []) {
      if (part?.text && !part?.thought) {
        const text = safeStreamText(span, accumulated, candidateIndex, part.text);
        retainStreamString(accumulated, text, (value) => {
          bucket.textParts.push(value);
        });
      } else if (part?.thought && part?.text) {
        const text = safeStreamText(span, accumulated, candidateIndex, part.text);
        retainStreamString(accumulated, text, (value) => {
          bucket.thinkingParts.push(value);
        });
      } else if (part?.functionCall) {
        const captured = captureMediaWithIndex(
          span,
          `neatlogs.llm.output_messages.${candidateIndex}`,
          part.functionCall.args,
          'output',
          accumulated.mediaCount,
        );
        accumulated.mediaCount += captured.count;
        retainStreamToolCall(accumulated, bucket, {
          ...part.functionCall,
          args: captured.value,
        });
      } else if (!part?.text) {
        const captured = captureMediaWithIndex(
          span,
          `neatlogs.llm.output_messages.${candidateIndex}`,
          part,
          'output',
          accumulated.mediaCount,
        );
        accumulated.mediaCount += captured.count;
      }
    }
    if (candidate?.finishReason) {
      bucket.finishReason = String(candidate.finishReason);
    }
  }
  if (chunk?.usageMetadata) accumulated.usage = chunk.usageMetadata;
  if (chunk?.promptFeedback?.blockReason) accumulated.promptFeedback = chunk.promptFeedback;
}

function finalizeStreamChunks(
  span: Span,
  accumulated: StreamAccumulator,
  cancelled = false,
): void {
  if (accumulated.finalized) return;
  accumulated.finalized = true;
  applyStreamChunks(span, accumulated);
  const blocked = accumulated.candidates.size === 0 ? accumulated.promptFeedback : undefined;
  if (blocked?.blockReason) recordBlockedPrompt(span, blocked);
  else if (cancelled) markCancelled(span);
  else span.setStatus({ code: SpanStatusCode.OK });
  span.end();
}

/** A broken stream keeps what it had already delivered. */
function failStreamChunks(span: Span, accumulated: StreamAccumulator, err: unknown): void {
  if (accumulated.finalized) return;
  accumulated.finalized = true;
  applyStreamChunks(span, accumulated);
  recordError(span, err);
}

/**
 * The API refused the prompt: no candidates, only `promptFeedback.blockReason`.
 * Nothing is thrown, so without this the call would end OK with no output.
 */
function recordBlockedPrompt(span: Span, promptFeedback: any): void {
  const reason = String(promptFeedback.blockReason);
  const detail = promptFeedback.blockReasonMessage;
  span.setAttribute('neatlogs.llm.finish_reason', reason);
  recordReportedFailure(
    span,
    'BlockedPrompt',
    detail ? `Prompt blocked: ${reason} (${detail})` : `Prompt blocked: ${reason}`,
  );
}

function applyStreamChunks(span: Span, accumulated: StreamAccumulator): void {
  const indexes = [...accumulated.candidates.keys()].sort((a, b) => a - b);
  let flattenedToolIndex = 0;
  for (const candidateIndex of indexes) {
    const bucket = accumulated.candidates.get(candidateIndex)!;
    const prefix = `neatlogs.llm.output_messages.${candidateIndex}`;
    const fullText = bucket.textParts.join('');
    const fullThinking = bucket.thinkingParts.join('');
    if (fullText) {
      span.setAttribute(`${prefix}.role`, 'assistant');
      span.setAttribute(`${prefix}.content`, fullText);
    }
    if (fullThinking) {
      span.setAttribute(`${prefix}.thinking`, fullThinking);
    }
    if (bucket.finishReason) {
      span.setAttribute(`neatlogs.llm.choices.${candidateIndex}.finish_reason`, bucket.finishReason);
      if (candidateIndex === indexes[0]) {
        span.setAttribute('neatlogs.llm.finish_reason', bucket.finishReason);
      }
    }
    for (let toolIndex = 0; toolIndex < bucket.toolCalls.length; toolIndex++) {
      const tool = bucket.toolCalls[toolIndex];
      const toolPrefix = `neatlogs.llm.tool_calls.${flattenedToolIndex}`;
      if (tool.id) span.setAttribute(`${toolPrefix}.id`, tool.id);
      span.setAttribute(`${toolPrefix}.name`, tool.name);
      span.setAttribute(`${toolPrefix}.arguments`, tool.arguments);
      span.setAttribute(`${toolPrefix}.choice_index`, candidateIndex);
      span.setAttribute(`${toolPrefix}.tool_call_index`, toolIndex);
      flattenedToolIndex++;
    }
  }
  setUsage(span, accumulated.usage);
  applyStreamDiagnostics(span, accumulated);
}

// ---------------------------------------------------------------------------
// Non-streaming response finalization
// ---------------------------------------------------------------------------

function finalizeResponse(span: Span, response: any): void {
  const candidates = response?.candidates ?? [];
  let flattenedToolIndex = 0;
  let firstCandidateIndex: number | null = null;

  for (let position = 0; position < candidates.length; position++) {
    const candidate = candidates[position];
    const candidateIndex = typeof candidate?.index === 'number' ? candidate.index : position;
    if (firstCandidateIndex === null) firstCandidateIndex = candidateIndex;
    const prefix = `neatlogs.llm.output_messages.${candidateIndex}`;
    const captured = captureMedia(span, prefix, [candidate], 'output');
    // Never fall back to the raw candidate: if sanitization failed or hit its
    // media limit, keep the typed media records already written and mark the
    // candidate as incomplete instead of serializing provider bytes.
    const sanitizedCandidate = Array.isArray(captured) ? captured[0] : undefined;
    if (!Array.isArray(captured)) {
      span.setAttribute(`${prefix}.media.incomplete`, true);
      span.setAttribute(`${prefix}.content`, String(captured));
    }

    const textParts: string[] = [];
    const thinkingParts: string[] = [];
    let toolIndex = 0;
    for (const part of sanitizedCandidate?.content?.parts ?? []) {
      if (part?.text && !part?.thought) {
        textParts.push(part.text);
      } else if (part?.thought && part?.text) {
        thinkingParts.push(part.text);
      } else if (part?.functionCall) {
        const fc = part.functionCall;
        const toolPrefix = `neatlogs.llm.tool_calls.${flattenedToolIndex}`;
        if (fc?.id) span.setAttribute(`${toolPrefix}.id`, String(fc.id));
        span.setAttribute(`${toolPrefix}.name`, fc?.name ?? '');
        span.setAttribute(`${toolPrefix}.arguments`, safeStringify(fc?.args ?? {}));
        span.setAttribute(`${toolPrefix}.choice_index`, candidateIndex);
        span.setAttribute(`${toolPrefix}.tool_call_index`, toolIndex);
        flattenedToolIndex++;
        toolIndex++;
      }
    }
    if (textParts.length) {
      span.setAttribute(`${prefix}.role`, 'assistant');
      span.setAttribute(`${prefix}.content`, textParts.join(''));
    }
    if (thinkingParts.length) {
      span.setAttribute(`${prefix}.thinking`, thinkingParts.join(''));
    }
    if (candidate?.finishReason) {
      span.setAttribute(`neatlogs.llm.choices.${candidateIndex}.finish_reason`, String(candidate.finishReason));
      if (candidateIndex === firstCandidateIndex) {
        span.setAttribute('neatlogs.llm.finish_reason', String(candidate.finishReason));
      }
    }
  }

  setUsage(span, response?.usageMetadata);

  if (candidates.length === 0 && response?.promptFeedback?.blockReason) {
    recordBlockedPrompt(span, response.promptFeedback);
  } else {
    span.setStatus({ code: SpanStatusCode.OK });
  }
  span.end();
}

function setUsage(span: Span, usage: any): void {
  if (!usage) return;
  if (usage.promptTokenCount != null) span.setAttribute('neatlogs.llm.token_count.prompt', usage.promptTokenCount);
  if (usage.candidatesTokenCount != null) span.setAttribute('neatlogs.llm.token_count.completion', usage.candidatesTokenCount);
  if (usage.totalTokenCount != null) span.setAttribute('neatlogs.llm.token_count.total', usage.totalTokenCount);
  if (usage.cachedContentTokenCount != null) span.setAttribute('neatlogs.llm.token_count.cache_read', usage.cachedContentTokenCount);
  if (usage.thoughtsTokenCount != null) span.setAttribute('neatlogs.llm.token_count.reasoning', usage.thoughtsTokenCount);
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function markStreamIncomplete(accumulated: StreamAccumulator, reason: string): void {
  accumulated.incompleteReasons.add(reason);
}

function retainStreamString(
  accumulated: StreamAccumulator,
  value: string,
  retain: (value: string) => void,
): boolean {
  const remaining = Math.max(
    0,
    DEFAULT_MAX_STREAM_CAPTURE_BYTES - accumulated.capturedBytes,
  );
  const bytes = utf8ByteLength(value, remaining);
  if (accumulated.capturedItems >= DEFAULT_MAX_STREAM_CAPTURE_ITEMS) {
    accumulated.droppedItems += 1;
    accumulated.droppedBytes += bytes;
    markStreamIncomplete(accumulated, 'item_limit_exceeded');
    return false;
  }
  if (accumulated.capturedBytes + bytes > DEFAULT_MAX_STREAM_CAPTURE_BYTES) {
    accumulated.droppedItems += 1;
    accumulated.droppedBytes += bytes;
    markStreamIncomplete(accumulated, 'byte_limit_exceeded');
    return false;
  }
  retain(value);
  accumulated.capturedItems += 1;
  accumulated.capturedBytes += bytes;
  return true;
}

function retainStreamToolCall(
  accumulated: StreamAccumulator,
  bucket: CandidateAccumulator,
  functionCall: any,
): void {
  const id = String(functionCall?.id ?? '');
  const name = String(functionCall?.name ?? '');
  const args = safeStringify(functionCall?.args ?? {});
  const retained = retainStreamString(
    accumulated,
    `${id}\0${name}\0${args}`,
    () => bucket.toolCalls.push({ id, name, arguments: args }),
  );
  if (!retained) markStreamIncomplete(accumulated, 'tool_call_omitted');
}

function applyStreamDiagnostics(span: Span, accumulated: StreamAccumulator): void {
  span.setAttribute('neatlogs.stream.capture_bytes', accumulated.capturedBytes);
  span.setAttribute('neatlogs.stream.capture_items', accumulated.capturedItems);
  if (accumulated.incompleteReasons.size === 0) return;
  span.setAttribute('neatlogs.stream.incomplete', true);
  span.setAttribute(
    'neatlogs.stream.incomplete_reason',
    [...accumulated.incompleteReasons].sort().join(','),
  );
  if (accumulated.droppedBytes > 0) {
    span.setAttribute('neatlogs.stream.dropped_bytes', accumulated.droppedBytes);
    span.setAttribute('neatlogs.stream.dropped_bytes_is_lower_bound', true);
  }
  span.setAttribute('neatlogs.stream.dropped_items', accumulated.droppedItems);
  span.setAttribute('neatlogs.capture_fidelity', 'truncated');
}

function safeStringify(value: unknown): string {
  try {
    return typeof value === 'string' ? value : JSON.stringify(value);
  } catch {
    return '';
  }
}

function recordError(span: Span, err: unknown): void {
  recordFailure(span, err);
  span.end();
}
