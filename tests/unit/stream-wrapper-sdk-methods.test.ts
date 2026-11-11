import { afterAll, beforeAll, beforeEach, describe, expect, it } from 'vitest';
import http from 'node:http';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { InMemorySpanExporter, SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import Anthropic from '@anthropic-ai/sdk';
import OpenAI, { AzureOpenAI } from 'openai';
import { SpanStatusCode } from '@opentelemetry/api';
import { _setNeatlogsProvider } from '../../src/core/provider.js';
import { wrapAnthropic } from '../../src/anthropic.js';
import { wrapAzureOpenAI } from '../../src/azure-openai.js';
import { wrapOpenAI } from '../../src/openai.js';

let exporter: InMemorySpanExporter;
let provider: NodeTracerProvider;
let previousAutoRoot: string | undefined;
let server: http.Server;
let baseUrl: string;

beforeAll(async () => {
  previousAutoRoot = process.env.NEATLOGS_AUTO_ROOT;
  process.env.NEATLOGS_AUTO_ROOT = 'false';
  exporter = new InMemorySpanExporter();
  provider = new NodeTracerProvider();
  provider.addSpanProcessor(new SimpleSpanProcessor(exporter));
  _setNeatlogsProvider(provider);

  server = http.createServer((request, response) => {
    response.writeHead(200, { 'content-type': 'text/event-stream' });
    const failMidStream = request.url?.startsWith('/error') === true;
    const write = (text: string) => response.write(text);
    if (request.url?.includes('/messages')) {
      const event = (name: string, data: unknown) =>
        write(`event: ${name}\ndata: ${JSON.stringify(data)}\n\n`);
      event('message_start', { type: 'message_start', message: { id: 'm1', type: 'message', role: 'assistant', model: 'claude-test', content: [], stop_reason: null, usage: { input_tokens: 7, output_tokens: 1 } } });
      event('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
      event('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'Hello' } });
      if (failMidStream) event('error', { type: 'error', error: { type: 'overloaded_error', message: 'upstream overloaded' } });
      event('content_block_stop', { type: 'content_block_stop', index: 0 });
      event('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { output_tokens: 5 } });
      event('message_stop', { type: 'message_stop' });
    } else if (request.url?.includes('/responses')) {
      const event = (data: { type: string }) =>
        write(`event: ${data.type}\ndata: ${JSON.stringify(data)}\n\n`);
      event({ type: 'response.created', response: { id: 'r1', status: 'in_progress', output: [] } });
      event({ type: 'response.output_text.delta', delta: 'Hel' });
      if (failMidStream) write(`data: ${JSON.stringify({ error: { message: 'upstream overloaded', type: 'server_error' } })}\n\n`);
      event({ type: 'response.output_text.delta', delta: 'lo' });
      event({ type: 'response.completed', response: { id: 'r1', status: 'completed', model: 'gpt-test', output_text: 'Hello', output: [{ type: 'message', id: 'i1', role: 'assistant', status: 'completed', content: [{ type: 'output_text', text: 'Hello', annotations: [] }] }], usage: { input_tokens: 3, output_tokens: 2, total_tokens: 5 } } });
    } else {
      const chunk = (choice: unknown) =>
        write(`data: ${JSON.stringify({ id: 'c1', object: 'chat.completion.chunk', model: 'gpt-test', choices: [choice] })}\n\n`);
      chunk({ index: 0, delta: { role: 'assistant', content: 'Hel' } });
      if (failMidStream) write(`data: ${JSON.stringify({ error: { message: 'upstream overloaded', type: 'server_error' } })}\n\n`);
      chunk({ index: 0, delta: { content: 'lo' } });
      chunk({ index: 0, delta: {}, finish_reason: 'stop' });
      write(`data: [DONE]\n\n`);
    }
    response.end();
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  baseUrl = `http://127.0.0.1:${(server.address() as { port: number }).port}`;
});
beforeEach(() => exporter.reset());
afterAll(async () => {
  server.close();
  _setNeatlogsProvider(null);
  await provider.shutdown();
  if (previousAutoRoot === undefined) delete process.env.NEATLOGS_AUTO_ROOT;
  else process.env.NEATLOGS_AUTO_ROOT = previousAutoRoot;
});

const streams = [
  [
    'anthropic',
    (path = '') =>
      (wrapAnthropic(new Anthropic({ apiKey: 'test', baseURL: baseUrl + path, maxRetries: 0 })) as any).messages.create({
        model: 'claude-test',
        max_tokens: 10,
        messages: [{ role: 'user', content: 'hi' }],
        stream: true,
      }),
  ],
  [
    'openai',
    (path = '') =>
      (wrapOpenAI(new OpenAI({ apiKey: 'test', baseURL: baseUrl + path, maxRetries: 0 })) as any).chat.completions.create({
        model: 'gpt-test',
        messages: [{ role: 'user', content: 'hi' }],
        stream: true,
      }),
  ],
  [
    'openai-responses',
    (path = '') =>
      (wrapOpenAI(new OpenAI({ apiKey: 'test', baseURL: baseUrl + path, maxRetries: 0 })) as any).responses.create({
        model: 'gpt-test',
        input: 'hi',
        stream: true,
      }),
  ],
  [
    'azure-openai',
    (path = '') =>
      (wrapAzureOpenAI(new AzureOpenAI({ apiKey: 'test', endpoint: baseUrl + path, apiVersion: '2024-10-21', deployment: 'test', maxRetries: 0 })) as any).chat.completions.create({
        model: 'gpt-test',
        messages: [{ role: 'user', content: 'hi' }],
        stream: true,
      }),
  ],
] as const;

describe.each(streams)('%s wrapped stream keeps the SDK stream methods working', (_name, open: (path?: string) => Promise<any>) => {
  it('tee() returns two readable branches instead of throwing', async () => {
    const stream = await open();
    const [left, right] = stream.tee();
    const count = async (branch: AsyncIterable<unknown>) => {
      let events = 0;
      for await (const _event of branch) events++;
      return events;
    };
    const [leftEvents, rightEvents] = await Promise.all([count(left), count(right)]);
    expect(leftEvents).toBeGreaterThan(0);
    expect(rightEvents).toBe(leftEvents);
    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].attributes['neatlogs.llm.output_messages.0.content']).toBe('Hello');
  });

  it('toReadableStream() still records the response', async () => {
    const stream = await open();
    const reader = stream.toReadableStream().getReader();
    for (;;) {
      const { done } = await reader.read();
      if (done) break;
    }
    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].attributes['neatlogs.llm.output_messages.0.content']).toBe('Hello');
  });

  it('breaking out of for await still ends the span as cancelled', async () => {
    const stream = await open();
    for await (const _event of stream) break;
    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].attributes['neatlogs.stream.cancelled']).toBe(true);
  });

  it('cancelling the readable stream ends the span as cancelled', async () => {
    const stream = await open();
    const reader = stream.toReadableStream().getReader();
    await reader.read();
    await reader.cancel();
    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].attributes['neatlogs.stream.cancelled']).toBe(true);
  });

  it('a mid-stream error reaches the reader unchanged and marks the span as an error', async () => {
    const stream = await open('/error');
    const failure = await new Response(stream.toReadableStream()).text().catch((err) => err);
    expect(failure.message).toContain('upstream overloaded');
    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].status.code).toBe(SpanStatusCode.ERROR);
  });

  it('a tee branch left unread does not block the other branch from ending the span', async () => {
    const stream = await open();
    const [left] = stream.tee();
    for await (const _event of left);
    expect(exporter.getFinishedSpans()).toHaveLength(1);
  });
});

describe('streams without an SDK iterator method', () => {
  const openaiChunks = [
    { id: 'c1', object: 'chat.completion.chunk', model: 'gpt-test', choices: [{ index: 0, delta: { role: 'assistant', content: 'Hello' } }] },
    { id: 'c1', object: 'chat.completion.chunk', model: 'gpt-test', choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] },
  ];
  const anthropicEvents = [
    { type: 'message_start', message: { id: 'm1', type: 'message', role: 'assistant', model: 'claude-test', content: [], stop_reason: null, usage: { input_tokens: 7, output_tokens: 1 } } },
    { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
    { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'Hello' } },
    { type: 'content_block_stop', index: 0 },
    { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { output_tokens: 5 } },
    { type: 'message_stop' },
  ];
  const plainStream = (items: unknown[]) => ({
    async *[Symbol.asyncIterator]() {
      yield* items;
    },
  });
  const fallbacks = [
    ['anthropic', () => (wrapAnthropic({ messages: { create: async () => plainStream(anthropicEvents) } } as any) as any).messages.create({ model: 'claude-test', max_tokens: 10, messages: [], stream: true })],
    ['openai', () => (wrapOpenAI({ chat: { completions: { create: async () => plainStream(openaiChunks) } } }) as any).chat.completions.create({ model: 'gpt-test', messages: [], stream: true })],
    ['azure-openai', () => (wrapAzureOpenAI({ chat: { completions: { create: async () => plainStream(openaiChunks) } } } as any) as any).chat.completions.create({ model: 'gpt-test', messages: [], stream: true })],
  ] as const;

  it.each(fallbacks)('%s still records a plain async iterable', async (_name, open) => {
    const stream = await open();
    for await (const _event of stream);
    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].attributes['neatlogs.llm.output_messages.0.content']).toBe('Hello');
  });
});
