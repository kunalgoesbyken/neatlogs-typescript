/**
 * Neatlogs Azure OpenAI wrapper — ES6 Proxy-based.
 *
 * Azure OpenAI is accessed through the `openai` SDK's `AzureOpenAI` client,
 * which exposes the same `chat.completions.create()` shape as `OpenAI`. This
 * wrapper traces those calls with provider `azure` so Azure traffic is
 * distinguished from direct OpenAI traffic.
 *
 * Usage:
 *   import { wrapAzureOpenAI } from 'neatlogs/azure-openai';
 *   import { AzureOpenAI } from 'openai';
 *   const client = wrapAzureOpenAI(new AzureOpenAI({ endpoint, apiKey, apiVersion }));
 *   await client.chat.completions.create({ model: 'gpt-4o', messages: [...] });
 *
 * Intercepts:
 *   - client.chat.completions.create() — non-streaming and streaming
 *   - client.responses.create()        — Responses API
 */

import { SpanStatusCode, type Span } from '@opentelemetry/api';
import { getProviderTracer } from './core/auto-root.js';
import { captureMedia } from './core/media.js';
import { ChoiceAccumulator } from './core/choice-accumulator.js';
import {
  finalizeResponsesResponse,
  wrapResponsesAsyncIterableStream,
} from './core/openai-responses.js';
import { recordFailure } from './core/span-outcome.js';
import { instrumentSdkStreamIterator } from './core/sdk-stream.js';
import {
  getNeatlogsTracer,
  getNeatlogsParentContext,
  withNeatlogsSpan,
} from './core/provider.js';

const TRACER_NAME = 'neatlogs.azure_openai';
const PROVIDER = 'azure';
const SYSTEM = 'azure';

export function wrapAzureOpenAI<T extends object>(client: T): T {
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

function wrapNamespace(target: any, path: string[]): any {
  return new Proxy(target, {
    get(obj, prop, receiver) {
      const value = Reflect.get(obj, prop, receiver);
      if (typeof prop === 'symbol' || String(prop).startsWith('_')) return value;

      const currentPath = [...path, String(prop)];
      const pathStr = currentPath.join('.');

      if (pathStr === 'chat.completions.create' && typeof value === 'function') {
        return tracedChatCompletionsCreate(value.bind(obj));
      }
      if (pathStr === 'responses.create' && typeof value === 'function') {
        return tracedResponsesCreate(value.bind(obj));
      }

      if (value && typeof value === 'object' && !Array.isArray(value) && isNamespace(currentPath)) {
        return wrapNamespace(value, currentPath);
      }

      return value;
    },
  });
}

function isNamespace(path: string[]): boolean {
  if (path.length > 3) return false;
  const key = path[path.length - 1];
  return ['chat', 'completions', 'responses', 'beta'].includes(key);
}

// ---------------------------------------------------------------------------
// chat.completions.create — non-streaming + streaming
// ---------------------------------------------------------------------------

function tracedChatCompletionsCreate(original: (...args: any[]) => any) {
  return function (opts: any, ...rest: any[]): any {
    const tracer = getProviderTracer(TRACER_NAME);
    const model = opts?.model ?? '';
    const messages: any[] = opts?.messages ?? [];
    const isStream = opts?.stream === true;

    const span = tracer.startSpan('azure_openai.chat.completions.create', {
      attributes: {
        'neatlogs.span.kind': 'LLM',
        'neatlogs.llm.provider': PROVIDER,
        'neatlogs.llm.system': SYSTEM,
        'neatlogs.llm.model_name': model,
        'neatlogs.llm.is_streaming': isStream,
      },
    }, getNeatlogsParentContext());

    for (let i = 0; i < messages.length; i++) {
      const msg = messages[i];
      span.setAttribute(`neatlogs.llm.input_messages.${i}.role`, msg.role ?? '');
      if (typeof msg.content === 'string') {
        span.setAttribute(`neatlogs.llm.input_messages.${i}.content`, msg.content);
      } else if (msg.content) {
        const captured = captureMedia(
          span,
          `neatlogs.llm.input_messages.${i}`,
          msg.content,
          'input',
        );
        span.setAttribute(`neatlogs.llm.input_messages.${i}.content`, safeStringify(captured));
      }
      if (msg.tool_call_id) {
        span.setAttribute(`neatlogs.llm.input_messages.${i}.tool_call_id`, msg.tool_call_id);
      }
    }

    if (opts?.tools) {
      for (let i = 0; i < opts.tools.length; i++) {
        const fn = opts.tools[i]?.function ?? {};
        span.setAttribute(`neatlogs.llm.tools.${i}.name`, fn.name ?? '');
        if (fn.description) span.setAttribute(`neatlogs.llm.tools.${i}.description`, fn.description);
        if (fn.parameters) span.setAttribute(`neatlogs.llm.tools.${i}.input_schema`, safeStringify(fn.parameters));
      }
    }

    setInvocationParams(span, opts);

    if (isStream) {
      opts = { ...opts };
      const streamOpts = opts.stream_options ?? {};
      if (!streamOpts.include_usage) {
        opts.stream_options = { ...streamOpts, include_usage: true };
      }
    }

    const promise = withNeatlogsSpan(span, () => original(opts, ...rest));

    return promise.then(
      (response: any) => {
        if (isStream) {
          return wrapAsyncIterableStream(response, span);
        }
        finalizeChatResponse(span, response);
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
// responses.create
// ---------------------------------------------------------------------------

function tracedResponsesCreate(original: (...args: any[]) => any) {
  return function (opts: any, ...rest: any[]): any {
    const tracer = getProviderTracer(TRACER_NAME);
    const model = opts?.model ?? '';
    const isStream = opts?.stream === true;

    const span = tracer.startSpan('azure_openai.responses.create', {
      attributes: {
        'neatlogs.span.kind': 'LLM',
        'neatlogs.llm.provider': PROVIDER,
        'neatlogs.llm.system': SYSTEM,
        'neatlogs.llm.model_name': model,
        'neatlogs.llm.is_streaming': isStream,
        'input.value': '',
      },
    }, getNeatlogsParentContext());

    const capturedInput = captureMedia(
      span,
      'neatlogs.llm.input_messages.0',
      opts?.input,
      'input',
    );
    span.setAttribute('input.value', safeStringify(capturedInput ?? ''));

    const promise = withNeatlogsSpan(span, () => original(opts, ...rest));

    return promise.then(
      (response: any) => {
        if (isStream) {
          return wrapResponsesAsyncIterableStream(response, span);
        }
        finalizeResponsesResponse(span, response);
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
// Streaming support
// ---------------------------------------------------------------------------

function wrapAsyncIterableStream(stream: any, span: Span): any {
  const accumulator = new ChoiceAccumulator();
  const safeSpan = mediaSafeAccumulatorSpan(span);
  let cancellationRequested = false;
  const originalAsyncIterator = stream?.[Symbol.asyncIterator]?.bind(stream);

  if (!originalAsyncIterator) {
    accumulator.addResponse(stream, span);
    accumulator.finish(safeSpan);
    return stream;
  }

  const wrapIterator = (iterator: AsyncIterator<any>): AsyncIterator<any> => {
    return {
      async next(): Promise<IteratorResult<any>> {
        try {
          const result = await iterator.next();
          if (result.done) {
            // The SDK ends the iterator quietly on an abort; a chunk with a
            // finish_reason means the reply did complete first.
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
          const result = await (iterator.return?.(value) ?? { done: true, value: undefined });
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

  if (instrumentSdkStreamIterator(stream, wrapIterator)) return stream;

  const wrapped = Object.create(Object.getPrototypeOf(stream));
  Object.assign(wrapped, stream);
  wrapped[Symbol.asyncIterator] = () => wrapIterator(originalAsyncIterator());

  return wrapped;
}

// ---------------------------------------------------------------------------
// Non-streaming response finalization
// ---------------------------------------------------------------------------

function finalizeChatResponse(span: Span, response: any): void {
  const choices = response?.choices ?? [];
  for (let i = 0; i < choices.length; i++) {
    const message = choices[i]?.message;
    if (!message) continue;

    span.setAttribute(`neatlogs.llm.output_messages.${i}.role`, 'assistant');
    if (message.content) {
      const captured = captureMedia(
        span,
        `neatlogs.llm.output_messages.${i}`,
        message.content,
        'output',
      );
      span.setAttribute(
        `neatlogs.llm.output_messages.${i}.content`,
        typeof captured === 'string' ? captured : safeStringify(captured),
      );
    }
    if (message.tool_calls) {
      for (let j = 0; j < message.tool_calls.length; j++) {
        const tc = message.tool_calls[j];
        span.setAttribute(`neatlogs.llm.tool_calls.${j}.id`, tc.id ?? '');
        span.setAttribute(`neatlogs.llm.tool_calls.${j}.name`, tc.function?.name ?? '');
        span.setAttribute(
          `neatlogs.llm.tool_calls.${j}.arguments`,
          sanitizedJsonToolArguments(
            span,
            `neatlogs.llm.tool_calls.${j}.arguments`,
            tc.function?.arguments ?? '{}',
          ),
        );
      }
    }
    if (choices[i].finish_reason) {
      span.setAttribute('neatlogs.llm.finish_reason', choices[i].finish_reason);
    }
  }

  const usage = response?.usage;
  if (usage) {
    if (usage.prompt_tokens != null) span.setAttribute('neatlogs.llm.token_count.prompt', usage.prompt_tokens);
    if (usage.completion_tokens != null) span.setAttribute('neatlogs.llm.token_count.completion', usage.completion_tokens);
    if (usage.total_tokens != null) span.setAttribute('neatlogs.llm.token_count.total', usage.total_tokens);
    if (usage.prompt_tokens_details?.cached_tokens != null) {
      span.setAttribute('neatlogs.llm.token_count.cache_read', usage.prompt_tokens_details.cached_tokens);
    }
    if (usage.completion_tokens_details?.reasoning_tokens != null) {
      span.setAttribute('neatlogs.llm.token_count.reasoning', usage.completion_tokens_details.reasoning_tokens);
    }
  }

  if (response?.model) span.setAttribute('neatlogs.llm.model_name', response.model);

  span.setStatus({ code: SpanStatusCode.OK });
  span.end();
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function sanitizedJsonToolArguments(
  span: Span,
  prefix: string,
  value: unknown,
): string {
  if (typeof value !== 'string') {
    return safeStringify(captureMedia(span, prefix, value, 'output'));
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(value);
  } catch {
    span.setAttribute('neatlogs.stream.incomplete', true);
    span.setAttribute('neatlogs.stream.incomplete_reason', 'invalid_tool_arguments');
    return '[incomplete: invalid tool arguments omitted]';
  }
  const original = safeStringify(parsed);
  const sanitized = safeStringify(captureMedia(span, prefix, parsed, 'output'));
  return original === sanitized ? value : sanitized;
}

function mediaSafeAccumulatorSpan(span: Span): Span {
  return new Proxy(span, {
    get(target, property) {
      if (property === 'setAttribute') {
        return (name: string, value: unknown) =>
          target.setAttribute(
            name,
            /^neatlogs\.llm\.tool_calls\.\d+\.arguments$/.test(name)
              ? sanitizedJsonToolArguments(target, name, value)
              : (value as any),
          );
      }
      const value = Reflect.get(target, property, target);
      return typeof value === 'function' ? value.bind(target) : value;
    },
  });
}

function setInvocationParams(span: Span, opts: any): void {
  if (opts?.temperature != null) span.setAttribute('neatlogs.llm.temperature', opts.temperature);
  if (opts?.top_p != null) span.setAttribute('neatlogs.llm.top_p', opts.top_p);
  if (opts?.max_tokens != null) span.setAttribute('neatlogs.llm.max_tokens', opts.max_tokens);
  if (opts?.frequency_penalty != null) span.setAttribute('neatlogs.llm.frequency_penalty', opts.frequency_penalty);
  if (opts?.presence_penalty != null) span.setAttribute('neatlogs.llm.presence_penalty', opts.presence_penalty);
  if (opts?.stop) span.setAttribute('neatlogs.llm.stop_sequences', safeStringify(opts.stop));
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
