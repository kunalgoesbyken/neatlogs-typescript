import { afterAll, beforeAll, beforeEach, describe, expect, it } from 'vitest';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
  type ReadableSpan,
} from '@opentelemetry/sdk-trace-base';

import {
  traceTool as traceAnthropicTool,
  wrapAnthropic,
} from '../../src/anthropic.js';
import { traceTool as traceBedrockTool, wrapBedrock } from '../../src/bedrock.js';
import {
  traceTool as traceGoogleTool,
  wrapGoogleGenAI,
} from '../../src/google-genai.js';
import {
  traceTool as traceVertexTool,
  wrapVertexAI,
} from '../../src/vertex-ai.js';
import { traceTool as traceOpenAITool, wrapOpenAI } from '../../src/openai.js';
import {
  traceTool as traceAzureTool,
  wrapAzureOpenAI,
} from '../../src/azure-openai.js';
import { discardPendingMedia } from '../../src/core/media.js';
import { _setNeatlogsProvider } from '../../src/core/provider.js';
import { utf8ByteLength } from '../../src/constants.js';

let provider: NodeTracerProvider;
let exporter: InMemorySpanExporter;
let previousAutoRoot: string | undefined;

beforeAll(() => {
  previousAutoRoot = process.env.NEATLOGS_AUTO_ROOT;
  process.env.NEATLOGS_AUTO_ROOT = 'false';
  exporter = new InMemorySpanExporter();
  provider = new NodeTracerProvider();
  provider.addSpanProcessor(new SimpleSpanProcessor(exporter));
  _setNeatlogsProvider(provider);
});

afterAll(async () => {
  _setNeatlogsProvider(null);
  await provider.shutdown();
  if (previousAutoRoot === undefined) delete process.env.NEATLOGS_AUTO_ROOT;
  else process.env.NEATLOGS_AUTO_ROOT = previousAutoRoot;
});

beforeEach(() => exporter.reset());

function span(): ReadableSpan {
  const spans = exporter.getFinishedSpans();
  expect(spans).toHaveLength(1);
  return spans[0];
}

function inlineImage(byteLength: number, fill = 0x61): any {
  return {
    type: 'image',
    source: {
      type: 'base64',
      media_type: 'image/png',
      data: Buffer.alloc(byteLength, fill).toString('base64'),
    },
  };
}

function expectNoRawMedia(readable: ReadableSpan, media: any): void {
  const raw = media.source.data as string;
  expect(JSON.stringify(readable.attributes)).not.toContain(raw.slice(0, 256));
  discardPendingMedia(readable as object);
}

describe('provider tool media safety', () => {
  it.each([new Error('service unavailable'), 'service unavailable'])('records Google tool failures without hiding thrown errors', async (failure) => {
    const traced = traceGoogleTool('lookup', async () => { throw failure; });
    await expect(traced({ id: 1 })).rejects.toBe(failure);
    const readable = span();
    expect(readable.status.code).toBe(2);
    expect(JSON.parse(String(readable.attributes['output.value']))).toEqual({ kind: 'tool_error', message: 'service unavailable' });
  });
  it.each([
    ['anthropic', traceAnthropicTool],
    ['bedrock', traceBedrockTool],
    ['google', traceGoogleTool],
    ['vertex', traceVertexTool],
    ['openai', traceOpenAITool],
    ['azure', traceAzureTool],
  ] as const)('%s traceTool serializes only captureMedia sanitized values', async (_name, traceTool) => {
    const input = { image: inlineImage(120_000, 0x31), label: 'input' };
    const output = { image: inlineImage(120_000, 0x32), label: 'output' };
    const traced = traceTool('inspect_media', async () => output);

    await traced(input);

    const readable = span();
    expect(String(readable.attributes['input.value'])).toContain('neatlogs_media');
    expect(String(readable.attributes['output.value'])).toContain('neatlogs_media');
    expect(readable.attributes['neatlogs.tool.input.media.0.mime_type']).toBe('image/png');
    expect(readable.attributes['neatlogs.tool.output.media.0.mime_type']).toBe('image/png');
    expectNoRawMedia(readable, input.image);
    expectNoRawMedia(readable, output.image);
  });

  it('does not duplicate a 5 MiB Anthropic tool argument and preserves later tool calls', async () => {
    const image = inlineImage(5 * 1024 * 1024, 0x41);
    const response = {
      model: 'claude-test',
      stop_reason: 'tool_use',
      content: [
        { type: 'tool_use', id: 'media-1', name: 'inspect', input: { image } },
        { type: 'tool_use', id: 'plain-2', name: 'lookup', input: { city: 'Pune' } },
      ],
    };
    const wrapped = wrapAnthropic({
      messages: { create: async () => response },
    } as any);

    expect(await (wrapped as any).messages.create({ model: 'claude-test', messages: [] })).toBe(response);

    const readable = span();
    expect(readable.attributes['neatlogs.llm.tool_calls.0.id']).toBe('media-1');
    expect(String(readable.attributes['neatlogs.llm.tool_calls.0.arguments'])).toContain('neatlogs_media');
    expect(readable.attributes['neatlogs.llm.tool_calls.1.id']).toBe('plain-2');
    expect(readable.attributes['neatlogs.llm.tool_calls.1.arguments']).toBe('{"city":"Pune"}');
    expectNoRawMedia(readable, image);
  });

  it.each([
    ['google', wrapGoogleGenAI],
    ['vertex', wrapVertexAI],
  ] as const)('%s never falls back to the raw candidate past the media limit', async (_name, wrap) => {
    // Match the SDK's 10,000-attribute span limit so the media records fit on the span.
    const limitedExporter = new InMemorySpanExporter();
    const limitedProvider = new NodeTracerProvider({ spanLimits: { attributeCountLimit: 10_000 } });
    limitedProvider.addSpanProcessor(new SimpleSpanProcessor(limitedExporter));
    _setNeatlogsProvider(limitedProvider);
    try {
      const images = Array.from({ length: 65 }, (_, index) =>
        Buffer.alloc(2_000, index + 1).toString('base64'),
      );
      const response = {
        candidates: [{
          content: { parts: [{ functionCall: {
            id: 'many',
            name: 'inspect',
            args: { images: images.map((data) => ({ inlineData: { mimeType: 'image/png', data } })) },
          } }] },
        }],
      };
      const wrapped = wrap({ models: { generateContent: async () => response } } as any);

      await (wrapped as any).models.generateContent({ model: 'gemini-test', contents: 'go' });

      const spans = limitedExporter.getFinishedSpans();
      expect(spans).toHaveLength(1);
      const readable = spans[0];
      const attributes = JSON.stringify(readable.attributes);
      for (const data of images) expect(attributes).not.toContain(data.slice(0, 256));
      expect(readable.attributes['neatlogs.llm.output_messages.0.media.incomplete']).toBe(true);
      expect(readable.attributes['neatlogs.llm.output_messages.0.media.63.type']).toBeDefined();
      expect(readable.attributes['neatlogs.llm.tool_calls.0.arguments']).toBeUndefined();
      discardPendingMedia(readable as object);
    } finally {
      _setNeatlogsProvider(provider);
      await limitedProvider.shutdown();
    }
  });

  it.each([
    ['google', wrapGoogleGenAI],
    ['vertex', wrapVertexAI],
  ] as const)('%s sanitizes non-streamed function-call arguments', async (_name, wrap) => {
    const data = Buffer.alloc(120_000, 0x43).toString('base64');
    const credentialUrl = 'https://user:password@example.com/file?token=secret#fragment';
    const response = {
      candidates: [{
        content: { parts: [{ functionCall: {
          id: 'unsafe',
          name: 'inspect',
          args: { image: { inlineData: { mimeType: 'image/png', data } }, credentialUrl },
        } }] },
      }],
    };
    const wrapped = wrap({ models: { generateContent: async () => response } } as any);

    expect(await (wrapped as any).models.generateContent({ model: 'gemini-test', contents: 'go' })).toBe(response);

    const readable = span();
    const args = String(readable.attributes['neatlogs.llm.tool_calls.0.arguments']);
    expect(args).toContain('neatlogs_media');
    expect(args).not.toContain(data.slice(0, 256));
    discardPendingMedia(readable as object);
  });

  it.each([
    ['google', wrapGoogleGenAI],
    ['vertex', wrapVertexAI],
  ] as const)('%s keeps multiple streamed function calls in order without raw media', async (_name, wrap) => {
    const image = {
      inlineData: {
        mimeType: 'image/png',
        data: Buffer.alloc(120_000, 0x42).toString('base64'),
      },
    };
    const stream = (async function* () {
      yield {
        candidates: [{
          content: {
            parts: [
              { functionCall: { id: 'a', name: 'inspect', args: { image } } },
              { functionCall: { id: 'b', name: 'lookup', args: { city: 'Delhi' } } },
            ],
          },
        }],
      };
    })();
    const wrapped = wrap({
      models: { generateContentStream: async () => stream },
    } as any);

    const result = await (wrapped as any).models.generateContentStream({
      model: 'gemini-test',
      contents: 'go',
    });
    for await (const _chunk of result) {
      // consume
    }

    const readable = span();
    expect(readable.attributes['neatlogs.llm.tool_calls.0.name']).toBe('inspect');
    expect(String(readable.attributes['neatlogs.llm.tool_calls.0.arguments'])).toContain('neatlogs_media');
    expect(readable.attributes['neatlogs.llm.tool_calls.1.name']).toBe('lookup');
    expect(readable.attributes['neatlogs.llm.tool_calls.1.arguments']).toBe('{"city":"Delhi"}');
    expectNoRawMedia(readable, { source: { data: image.inlineData.data } });
  });
});

describe('streamed thinking and text media', () => {
  it.each([
    ['google', wrapGoogleGenAI],
    ['vertex', wrapVertexAI],
  ] as const)('%s keeps data URLs out of streamed thinking and text', async (_name, wrap) => {
    const data = Buffer.alloc(120_000, 0x43).toString('base64');
    const url = `data:image/png;base64,${data}`;
    const stream = (async function* () {
      yield {
        candidates: [{
          content: { parts: [
            { thought: true, text: url },
            { text: url },
          ] },
        }],
      };
    })();
    const wrapped = wrap({
      models: { generateContentStream: async () => stream },
    } as any);
    const result = await (wrapped as any).models.generateContentStream({
      model: 'gemini-test',
      contents: 'go',
    });
    for await (const _chunk of result) {
      // consume
    }
    const readable = span();
    expect(JSON.stringify(readable.attributes)).not.toContain(data.slice(0, 256));
    discardPendingMedia(readable as object);
  });
});

describe('provider streaming capture bounds', () => {
  it('short-circuits UTF-8 byte counting at the requested limit', () => {
    expect(utf8ByteLength('é🙂', 2)).toBe(3);
    expect(utf8ByteLength('é🙂')).toBe(6);
  });

  it('marks Anthropic capture incomplete at the byte bound without changing yielded chunks', async () => {
    const first = 'a'.repeat(600_000);
    const second = 'b'.repeat(600_000);
    const events = [
      { type: 'content_block_delta', delta: { type: 'text_delta', text: first } },
      { type: 'content_block_delta', delta: { type: 'text_delta', text: second } },
    ];
    const wrapped = wrapAnthropic({
      messages: { create: async () => (async function* () { yield* events; })() },
    } as any);

    const stream = await (wrapped as any).messages.create({
      model: 'claude-test',
      messages: [],
      stream: true,
    });
    const yielded: any[] = [];
    for await (const event of stream) yielded.push(event);

    expect(yielded).toEqual(events);
    const readable = span();
    expect(readable.attributes['neatlogs.llm.output_messages.0.content']).toBe(first);
    expect(readable.attributes['neatlogs.stream.incomplete']).toBe(true);
    expect(String(readable.attributes['neatlogs.stream.incomplete_reason'])).toContain('byte_limit_exceeded');
    expect(Number(readable.attributes['neatlogs.stream.dropped_bytes'])).toBeGreaterThan(0);
    expect(readable.attributes['neatlogs.stream.dropped_bytes_is_lower_bound']).toBe(true);
  });

  it('marks Anthropic capture incomplete at the item bound', async () => {
    const wrapped = wrapAnthropic({
      messages: {
        create: async () => (async function* () {
          for (let i = 0; i < 1025; i++) {
            yield { type: 'content_block_delta', delta: { type: 'text_delta', text: 'x' } };
          }
        })(),
      },
    } as any);

    const stream = await (wrapped as any).messages.create({
      model: 'claude-test',
      messages: [],
      stream: true,
    });
    let count = 0;
    for await (const _event of stream) count++;

    expect(count).toBe(1025);
    const readable = span();
    expect(String(readable.attributes['neatlogs.llm.output_messages.0.content'])).toHaveLength(1024);
    expect(String(readable.attributes['neatlogs.stream.incomplete_reason'])).toContain('item_limit_exceeded');
    expect(readable.attributes['neatlogs.stream.dropped_items']).toBe(1);
  });

  it('sanitizes Anthropic tool JSON assembled across chunks and preserves multiple tools', async () => {
    const image = inlineImage(120_000, 0x45);
    const mediaArguments = JSON.stringify({ image });
    const split = Math.floor(mediaArguments.length / 2);
    const events = [
      { type: 'content_block_start', content_block: { type: 'tool_use', id: 'one', name: 'inspect', input: {} } },
      { type: 'content_block_delta', delta: { type: 'input_json_delta', partial_json: mediaArguments.slice(0, split) } },
      { type: 'content_block_delta', delta: { type: 'input_json_delta', partial_json: mediaArguments.slice(split) } },
      { type: 'content_block_stop' },
      { type: 'content_block_start', content_block: { type: 'tool_use', id: 'two', name: 'lookup', input: {} } },
      { type: 'content_block_delta', delta: { type: 'input_json_delta', partial_json: '{"city":"Chennai"}' } },
      { type: 'content_block_stop' },
    ];
    const wrapped = wrapAnthropic({
      messages: { create: async () => (async function* () { yield* events; })() },
    } as any);
    const stream = await (wrapped as any).messages.create({
      model: 'claude-test',
      messages: [],
      stream: true,
    });
    for await (const _event of stream) {
      // consume
    }

    const readable = span();
    expect(readable.attributes['neatlogs.llm.tool_calls.0.id']).toBe('one');
    expect(String(readable.attributes['neatlogs.llm.tool_calls.0.arguments'])).toContain('neatlogs_media');
    expect(readable.attributes['neatlogs.llm.tool_calls.1.id']).toBe('two');
    expect(readable.attributes['neatlogs.llm.tool_calls.1.arguments']).toBe('{"city":"Chennai"}');
    expectNoRawMedia(readable, image);
  });

  it('defers Anthropic finalization when iterator.return() yields cleanup content', async () => {
    const wrapped = wrapAnthropic({
      messages: {
        create: async () => (async function* () {
          try {
            yield { type: 'content_block_delta', delta: { type: 'text_delta', text: 'main' } };
          } finally {
            yield { type: 'content_block_delta', delta: { type: 'text_delta', text: '-cleanup' } };
          }
        })(),
      },
    } as any);
    const stream = await (wrapped as any).messages.create({
      model: 'claude-test',
      messages: [],
      stream: true,
    });
    const iterator = stream[Symbol.asyncIterator]();

    expect((await iterator.next()).value.delta.text).toBe('main');
    const cleanup = await iterator.return();
    expect(cleanup.done).toBe(false);
    expect(cleanup.value.delta.text).toBe('-cleanup');
    expect(exporter.getFinishedSpans()).toHaveLength(0);
    expect((await iterator.next()).done).toBe(true);

    const readable = span();
    expect(readable.attributes['neatlogs.llm.output_messages.0.content']).toBe('main-cleanup');
    expect(readable.attributes['neatlogs.stream.cancelled']).toBe(true);
    expect(String(readable.attributes['neatlogs.stream.incomplete_reason'])).toContain('consumer_cancelled');
  });

  it.each([
    ['google', wrapGoogleGenAI],
    ['vertex', wrapVertexAI],
  ] as const)('%s enforces the stream byte bound and reports omitted content', async (_name, wrap) => {
    const first = 'a'.repeat(700_000);
    const second = 'b'.repeat(700_000);
    const wrapped = wrap({
      models: {
        generateContentStream: async () => (async function* () {
          yield { candidates: [{ content: { parts: [{ text: first }] } }] };
          yield { candidates: [{ content: { parts: [{ text: second }] } }] };
        })(),
      },
    } as any);
    const stream = await (wrapped as any).models.generateContentStream({
      model: 'gemini-test',
      contents: 'go',
    });
    let chunks = 0;
    for await (const _chunk of stream) chunks++;

    expect(chunks).toBe(2);
    const readable = span();
    expect(readable.attributes['neatlogs.llm.output_messages.0.content']).toBe(first);
    expect(String(readable.attributes['neatlogs.stream.incomplete_reason'])).toContain('byte_limit_exceeded');
    expect(Number(readable.attributes['neatlogs.stream.dropped_bytes'])).toBeGreaterThan(0);
    expect(readable.attributes['neatlogs.stream.dropped_bytes_is_lower_bound']).toBe(true);
  });

  it('bounds Bedrock ConverseStream retention while yielding every provider event', async () => {
    class ConverseStreamCommand {
      constructor(public input: any) {}
    }
    const first = 'a'.repeat(700_000);
    const second = 'b'.repeat(700_000);
    const events = [
      { contentBlockDelta: { contentBlockIndex: 0, delta: { text: first } } },
      { contentBlockDelta: { contentBlockIndex: 0, delta: { text: second } } },
    ];
    const wrapped = wrapBedrock({
      send: async () => ({ stream: (async function* () { yield* events; })() }),
    } as any);
    const response = await (wrapped as any).send(new ConverseStreamCommand({ modelId: 'anthropic.test' }));
    const yielded: any[] = [];
    for await (const event of response.stream) yielded.push(event);

    expect(yielded).toEqual(events);
    const readable = span();
    expect(readable.attributes['neatlogs.llm.output_messages.0.content']).toBe(first);
    expect(String(readable.attributes['neatlogs.stream.incomplete_reason'])).toContain('byte_limit_exceeded');
  });

  it('bounds OpenAI Responses text retention and still yields all events', async () => {
    const first = 'a'.repeat(700_000);
    const second = 'b'.repeat(700_000);
    const events = [
      { type: 'response.output_text.delta', delta: first },
      { type: 'response.output_text.delta', delta: second },
      { type: 'response.completed', response: { model: 'gpt-test', status: 'completed' } },
    ];
    const wrapped = wrapOpenAI({
      responses: { create: async () => (async function* () { yield* events; })() },
    } as any);
    const stream = await (wrapped as any).responses.create({ model: 'gpt-test', input: 'go', stream: true });
    const yielded: any[] = [];
    for await (const event of stream) yielded.push(event);

    expect(yielded).toEqual(events);
    const readable = span();
    expect(readable.attributes['output.value']).toBe(first);
    expect(String(readable.attributes['neatlogs.stream.incomplete_reason'])).toContain('byte_limit_exceeded');
  });
});

describe('Bedrock tool-call media safety', () => {
  it('sanitizes Converse tool arguments and keeps multiple calls ordered', async () => {
    class ConverseCommand {
      constructor(public input: any) {}
    }
    const image = inlineImage(120_000, 0x44);
    const response = {
      output: {
        message: {
          content: [
            { toolUse: { toolUseId: 'one', name: 'inspect', input: { image } } },
            { toolUse: { toolUseId: 'two', name: 'lookup', input: { city: 'Kolkata' } } },
          ],
        },
      },
    };
    const wrapped = wrapBedrock({ send: async () => response } as any);

    expect(await (wrapped as any).send(new ConverseCommand({ modelId: 'anthropic.test' }))).toBe(response);

    const readable = span();
    expect(String(readable.attributes['neatlogs.llm.tool_calls.0.arguments'])).toContain('neatlogs_media');
    expect(readable.attributes['neatlogs.llm.tool_calls.1.id']).toBe('two');
    expect(readable.attributes['neatlogs.llm.tool_calls.1.arguments']).toBe('{"city":"Kolkata"}');
    expectNoRawMedia(readable, image);
  });
});

describe('OpenAI-compatible tool argument safety', () => {
  it.each([
    ['openai', wrapOpenAI],
    ['azure', wrapAzureOpenAI],
  ] as const)('%s sanitizes JSON tool arguments and preserves normal multi-tool fidelity', async (_name, wrap) => {
    const image = inlineImage(120_000, 0x43);
    const mediaArguments = JSON.stringify({ image });
    const response = {
      model: 'gpt-test',
      choices: [{
        message: {
          role: 'assistant',
          tool_calls: [
            { id: 'one', type: 'function', function: { name: 'inspect', arguments: mediaArguments } },
            { id: 'two', type: 'function', function: { name: 'lookup', arguments: '{ "city": "Mumbai" }' } },
          ],
        },
        finish_reason: 'tool_calls',
      }],
    };
    const wrapped = wrap({
      chat: { completions: { create: async () => response } },
    } as any);

    expect(await (wrapped as any).chat.completions.create({ model: 'gpt-test', messages: [] })).toBe(response);

    const readable = span();
    expect(String(readable.attributes['neatlogs.llm.tool_calls.0.arguments'])).toContain('neatlogs_media');
    expect(readable.attributes['neatlogs.llm.tool_calls.1.id']).toBe('two');
    expect(readable.attributes['neatlogs.llm.tool_calls.1.arguments']).toBe('{ "city": "Mumbai" }');
    expectNoRawMedia(readable, image);
  });
});
