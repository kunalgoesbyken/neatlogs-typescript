import { createServer, type Server } from 'node:http';
import { afterEach, describe, expect, it } from 'vitest';
import OpenAI from 'openai';
import { GenerateContentResponse, GoogleGenAI } from '@google/genai';
import { wrapOpenAI } from '../../src/openai.js';
import { wrapGoogleGenAI } from '../../src/google-genai.js';
import { ReplaySession } from '../../src/replay/session.js';
import { replayRecordingFromCase } from '../../src/replay/frozen-case.js';
import { MODEL_CAPTURE_ATTRIBUTE, readProviderCapture } from '../../src/replay/provider-capture.js';
import { UnifiedAttributeProcessor } from '../../src/core/attribute-processor.js';
import { AttributeMapper } from '../../src/config/attribute-mapper.js';
import { useSpans } from '../unit/failed-calls/harness.js';

const { only, spans } = useSpans();
const servers = new Set<Server>();
afterEach(async () => {
  for (const server of servers) {
    await new Promise<void>((resolve, reject) => {
      server.close(error => error ? reject(error) : resolve());
      server.closeAllConnections();
    });
  }
  servers.clear();
});
async function endpoint(responseBody: unknown) {
  let requests = 0;
  const server = createServer((request, response) => {
    requests++;
    request.resume();
    response.writeHead(200, { 'content-type': 'application/json', 'x-request-id': 'transport-only-id', 'x-private-test': 'transport-only-secret' });
    response.end(JSON.stringify(responseBody));
  });
  servers.add(server);
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('Missing local test address.');
  return { url: `http://127.0.0.1:${address.port}`, requests: () => requests };
}
function savedSession() {
  const span = only();
  const raw = span.attributes[MODEL_CAPTURE_ATTRIBUTE];
  expect(raw).toBeTypeOf('string');
  expect(raw).not.toContain('transport-only');
  expect(raw).not.toContain('local-test-api-key');
  const attributes = new UnifiedAttributeProcessor(new AttributeMapper()).normalize({
    name: span.name, kind: span.kind, trace_id: span.spanContext().traceId, span_id: span.spanContext().spanId,
    start_time: 1, end_time: 2, attributes: span.attributes,
  });
  expect(attributes[MODEL_CAPTURE_ATTRIBUTE]).toBe(raw);
  const { recording, skippedTools } = replayRecordingFromCase({ source: { traceId: span.spanContext().traceId }, context: { spans: [{
    span_id: span.spanContext().spanId, span_type: 'LLM', attributes, status: span.status,
  }] } });
  expect(recording.entries).toHaveLength(1);
  expect(skippedTools).toBe(0);
  recording.entries.push({ spanId: 'checkpoint', kind: 'tool', name: 'lookup', input: {}, output: null });
  return { session: new ReplaySession(recording, { mode: 'until_span', untilSpan: 'checkpoint' }), capture: readProviderCapture(raw) };
}

describe('capture to frozen-case provider replay over HTTP', () => {
  it('preserves OpenAI text, tool calls and usage from the real SDK without a second request', async () => {
    const body = { id: 'chat-saved', object: 'chat.completion', created: 123, model: 'local-model',
      choices: [{ index: 0, finish_reason: 'tool_calls', message: { role: 'assistant', content: 'Working',
        tool_calls: [{ id: 'call-1', type: 'function', function: { name: 'lookup', arguments: '{"id":1}' } }] } }],
      usage: { prompt_tokens: 11, completion_tokens: 4, total_tokens: 15 } };
    const server = await endpoint(body);
    const client = wrapOpenAI(new OpenAI({ apiKey: 'local-test-api-key', baseURL: server.url, maxRetries: 0 }));
    const input: OpenAI.ChatCompletionCreateParamsNonStreaming = { model: 'local-model', messages: [{ role: 'user', content: 'Hello' }] };
    const live = await client.chat.completions.create(input);
    const { session, capture } = savedSession();
    expect(capture.input).toEqual(input);
    expect(capture.output).toEqual(body);
    expect(await session.run(() => client.chat.completions.create(input))).toEqual(live);
    expect(server.requests()).toBe(1);
    expect(session.getStatus().hits).toBe(1);
    expect(spans()[1]?.attributes['neatlogs.llm.token_count.total']).toBe(0);
  });

  it('preserves the real Responses SDK output_text alongside raw output items', async () => {
    const body = { id: 'response-saved', object: 'response', created_at: 123, model: 'local-model', status: 'completed',
      output: [{ id: 'message-1', type: 'message', role: 'assistant', status: 'completed', content: [{ type: 'output_text', text: 'Saved', annotations: [] }] }],
      usage: { input_tokens: 9, output_tokens: 2, total_tokens: 11 } };
    const server = await endpoint(body);
    const client = wrapOpenAI(new OpenAI({ apiKey: 'local-test-api-key', baseURL: server.url, maxRetries: 0 }));
    const input = { model: 'local-model', input: 'Hello' };
    const live = await client.responses.create(input);
    expect(live.output_text).toBe('Saved');
    const { session, capture } = savedSession();
    expect(capture.output).toEqual({ ...body, output_text: 'Saved' });
    expect(await session.run(() => client.responses.create(input))).toEqual(live);
    expect(server.requests()).toBe(1);
  });

  it('restores the actual Google response fields and getters without retaining HTTP headers', async () => {
    const body = { responseId: 'google-saved', modelVersion: 'gemini-local',
      candidates: [{ content: { role: 'model', parts: [{ text: 'Saved' }, { functionCall: { name: 'lookup', args: { id: 1 } } }] }, finishReason: 'STOP', index: 0 }],
      usageMetadata: { promptTokenCount: 12, candidatesTokenCount: 3, totalTokenCount: 15 } };
    const server = await endpoint(body);
    const client = wrapGoogleGenAI(new GoogleGenAI({ apiKey: 'local-test-api-key', httpOptions: { baseUrl: server.url } }));
    const input = { model: 'gemini-local', contents: 'Hello' };
    const live = await client.models.generateContent(input);
    expect(live).toBeInstanceOf(GenerateContentResponse);
    const { session, capture } = savedSession();
    expect(capture.output).toEqual(body);
    const cached = await session.run(() => client.models.generateContent(input));
    expect(cached).toBeInstanceOf(GenerateContentResponse);
    expect(cached.text).toBe(live.text);
    expect(cached.functionCalls).toEqual(live.functionCalls);
    expect(cached.usageMetadata).toEqual(live.usageMetadata);
    expect(cached.sdkHttpResponse).toBeUndefined();
    expect(server.requests()).toBe(1);
  });
});
