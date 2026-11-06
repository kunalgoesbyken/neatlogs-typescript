import { describe, expect, it, vi } from 'vitest';
import { GenerateContentResponse } from '@google/genai';
import OpenAI from 'openai';
import { providerReplayInput } from '../../src/replay/provider.js';
import { wrapOpenAI } from '../../src/openai.js';
import { wrapGoogleGenAI } from '../../src/google-genai.js';
import { ReplaySession } from '../../src/replay/session.js';
import type { ReplayEntry } from '../../src/replay/contracts.js';
import { useSpans } from '../unit/failed-calls/harness.js';

const { only } = useSpans();
const request = { model: 'test-model', messages: [{ role: 'user', content: 'Hello' }] };
const answer = { id: 'completion-saved', object: 'chat.completion', created: 123, model: 'test-model',
  choices: [{ index: 0, finish_reason: 'stop', message: { role: 'assistant', content: 'Saved' } }],
  usage: { prompt_tokens: 100, completion_tokens: 10, total_tokens: 110 } };
const checkpoint: ReplayEntry = { spanId: 'checkpoint', kind: 'tool', name: 'lookup', input: {}, output: null };
function session(name: string, input: unknown, output: unknown, mode: 'until_span' | 'tools_only' = 'until_span') {
  return new ReplaySession({ version: 1, sourceTraceId: 'source', entries: [
    { spanId: 'saved-model', kind: 'model', name, input, output }, checkpoint,
  ] }, { mode, ...(mode === 'until_span' ? { untilSpan: checkpoint.spanId } : {}) });
}

describe('provider-backed local replay', () => {
  it('returns the complete saved OpenAI response without a request and records zero newly billed tokens', async () => {
    const create = vi.fn(async (_request: unknown) => ({ ...answer, id: 'live' }));
    const client = wrapOpenAI({ chat: { completions: { create } } });
    const replay = session('openai.chat.completions.create', request, answer);
    const response = await replay.run(() => client.chat.completions.create({ ...request, temperature: undefined }));
    expect(response).toEqual(answer); expect(create).not.toHaveBeenCalled();
    expect(replay.getStatus()).toMatchObject({ hits: 1, liveCalls: 0, switchedToLive: false });
    expect(only().attributes).toMatchObject({ 'neatlogs.replay.outcome': 'HIT', 'neatlogs.replay.source_span_id': 'saved-model',
      'neatlogs.llm.token_count.prompt': 0, 'neatlogs.llm.token_count.completion': 0, 'neatlogs.llm.token_count.total': 0 });
    // Application code still sees the recorded provider usage, unchanged.
    expect(response.usage.total_tokens).toBe(110);
  });

  it.each(['openai', 'google'])('matches nested undefined fields for %s and still misses changed values', async provider => {
    const google = { model: 'gemini-test', contents: [{ role: 'user', parts: [{ text: 'Hello' }] }], config: { temperature: .2 } };
    const input = provider === 'openai' ? request : google;
    const saved = provider === 'openai' ? answer : { candidates: [{ content: { parts: [{ text: 'Saved' }] } }] };
    const live = vi.fn(async () => saved);
    const client = provider === 'openai' ? wrapOpenAI({ chat: { completions: { create: live } } }) : wrapGoogleGenAI({ models: { generateContent: live } });
    const call = (value: unknown) => provider === 'openai' ? client.chat.completions.create(value) : client.models.generateContent(value);
    const replay = session(provider === 'openai' ? 'openai.chat.completions.create' : 'google_genai.models.generate_content', input, saved);
    const equivalent = provider === 'openai' ? { ...request, messages: [{ ...request.messages[0], name: undefined }] } : { ...google, config: { ...google.config, seed: undefined } };
    await replay.run(() => call(equivalent));
    expect(live).not.toHaveBeenCalled();
    expect(replay.getStatus().hits).toBe(1);
    const changed = session(provider === 'openai' ? 'openai.chat.completions.create' : 'google_genai.models.generate_content', input, saved);
    await changed.run(() => call({ ...equivalent, model: 'different' }));
    expect(live).toHaveBeenCalledOnce();
    expect(changed.getStatus().reason).toBe('missing_recording');
  });

  it('keeps invalid array elements, computed fields and unsupported values out of replay matching', () => {
    for (const input of [{ messages: [undefined] }, { options: { seed: BigInt(1) } }, { options: new Date() }])
      expect(() => providerReplayInput(input)).toThrow();
    const getter = vi.fn(() => 'value');
    const input = { model: 'test' };
    Object.defineProperty(input, 'config', { get: getter, enumerable: true });
    expect(() => providerReplayInput(input)).toThrow();
    expect(getter).not.toHaveBeenCalled();
  });

  it.each(['chat', 'responses'].flatMap(operation => ['withResponse', 'asResponse'].map(helper => [operation, helper])))('uses a recorded live miss for %s.%s HTTP metadata', async (operation, helper) => {
    const input = operation === 'chat' ? request : { model: 'test-model', input: 'Hello' };
    const saved = operation === 'chat' ? answer : { id: 'response-saved', object: 'response', created_at: 123, model: 'test-model', status: 'completed', output: [], output_text: 'Saved' };
    const response = Response.json({ ...saved, id: 'live' }, { headers: { 'x-request-id': 'actual-request' } });
    const live = vi.fn(async () => response);
    const client = wrapOpenAI(new OpenAI({ apiKey: 'test-provider', baseURL: 'https://models.example.invalid/v1', fetch: live, maxRetries: 0 }));
    const replay = session(operation === 'chat' ? 'openai.chat.completions.create' : 'openai.responses.create', input, saved);
    const result = await replay.run(() => (operation === 'chat' ? client.chat.completions.create(input) : client.responses.create(input))[helper]());
    expect(helper === 'withResponse' ? result.response : result).toBe(response);
    if (helper === 'withResponse') expect(result.data.id).toBe('live');
    expect(live).toHaveBeenCalledOnce();
    expect(replay.getStatus()).toMatchObject({ hits: 0, liveCalls: 1, reason: 'missing_recording' });
  });

  it.each(['tools_only', 'missing'] as const)('keeps known %s calls eager and concurrent with the installed OpenAI client', async mode => {
    let active = 0, concurrent = 0;
    const live = vi.fn(async () => {
      concurrent = Math.max(concurrent, ++active);
      await new Promise(resolve => setTimeout(resolve, 20));
      active--;
      return Response.json(answer);
    });
    const client = wrapOpenAI(new OpenAI({ apiKey: 'test-provider', baseURL: 'https://models.example.invalid/v1', fetch: live, maxRetries: 0 }));
    const replay = session('openai.chat.completions.create', request, answer, mode === 'tools_only' ? mode : 'until_span');
    await replay.run(async () => {
      const first = client.chat.completions.create({ ...request, model: 'new-model' });
      const second = client.chat.completions.create({ ...request, model: 'new-model' });
      await first;
      await second;
    });
    expect(live).toHaveBeenCalledTimes(2);
    expect(concurrent).toBe(2);
    expect(replay.getStatus()).toMatchObject({ hits: 0, liveCalls: 2, switchedToLive: mode === 'missing' });
  });

  it('reserves duplicate requests in invocation order even when awaited in reverse', async () => {
    const live = vi.fn(async () => answer);
    const client = wrapOpenAI({ chat: { completions: { create: live } } });
    const replay = new ReplaySession({ version: 1, sourceTraceId: 'source', entries: [
      { spanId: 'first', kind: 'model', name: 'openai.chat.completions.create', input: request, output: { ...answer, id: 'first' } },
      { spanId: 'second', kind: 'model', name: 'openai.chat.completions.create', input: request, output: { ...answer, id: 'second' } }, checkpoint,
    ] }, { mode: 'until_span', untilSpan: checkpoint.spanId });
    await replay.run(async () => {
      const first = client.chat.completions.create(request), second = client.chat.completions.create(request);
      expect(replay.getStatus()).toMatchObject({ hits: 0, liveCalls: 0 });
      expect((await second).id).toBe('second');
      expect((await first).id).toBe('first');
    });
    expect(live).not.toHaveBeenCalled();
    expect(replay.getStatus()).toMatchObject({ hits: 2, liveCalls: 0 });
  });

  it.each(['withResponse', 'asResponse'])('rejects late %s without undoing a later consumed HIT or making a paid request', async helper => {
    const live = vi.fn(async () => answer);
    const client = wrapOpenAI({ chat: { completions: { create: live } } });
    const replay = new ReplaySession({ version: 1, sourceTraceId: 'source', entries: [
      { spanId: 'first', kind: 'model', name: 'openai.chat.completions.create', input: request, output: answer },
      { spanId: 'second', kind: 'model', name: 'openai.chat.completions.create', input: request, output: answer }, checkpoint,
    ] }, { mode: 'until_span', untilSpan: checkpoint.spanId });
    await replay.run(async () => {
      const first = client.chat.completions.create(request), second = client.chat.completions.create(request);
      await second;
      await expect(first[helper]()).rejects.toThrow(/before consuming a later/);
    });
    expect(live).not.toHaveBeenCalled();
    expect(replay.getStatus()).toMatchObject({ hits: 1, liveCalls: 0, switchedToLive: false });
  });

  it('a first raw helper invalidates later unconsumed reservations and counts actual live calls', async () => {
    const live = vi.fn(() => Object.assign(Promise.resolve({ ...answer, id: 'live' }), { asResponse: async () => Response.json(answer) }));
    const client = wrapOpenAI({ chat: { completions: { create: live } } });
    const replay = new ReplaySession({ version: 1, sourceTraceId: 'source', entries: [
      { spanId: 'first', kind: 'model', name: 'openai.chat.completions.create', input: request, output: answer },
      { spanId: 'second', kind: 'model', name: 'openai.chat.completions.create', input: request, output: answer }, checkpoint,
    ] }, { mode: 'until_span', untilSpan: checkpoint.spanId });
    await replay.run(async () => {
      const first = client.chat.completions.create(request), second = client.chat.completions.create(request);
      await first.withResponse();
      expect((await second).id).toBe('live');
    });
    expect(live).toHaveBeenCalledTimes(2);
    expect(replay.getStatus()).toMatchObject({ hits: 0, liveCalls: 2, reason: 'missing_recording' });
  });

  it('does not charge a new request for a raw helper requested after consuming a saved response', async () => {
    const live = vi.fn(async () => answer);
    const client = wrapOpenAI({ chat: { completions: { create: live } } });
    const replay = session('openai.chat.completions.create', request, answer);
    await replay.run(async () => {
      const response = client.chat.completions.create(request);
      expect(await response).toEqual(answer);
      await expect(response.asResponse()).rejects.toThrow(/before consuming/);
    });
    expect(live).not.toHaveBeenCalled();
    expect(replay.getStatus()).toMatchObject({ hits: 1, liveCalls: 0 });
  });

  it('executes the checkpoint model and all following calls live', async () => {
    const create = vi.fn(async (_request: unknown) => ({ ...answer, id: 'live' }));
    const client = wrapOpenAI({ chat: { completions: { create } } });
    const replay = new ReplaySession({ version: 1, sourceTraceId: 'source', entries: [
      { spanId: 'first', kind: 'model', name: 'openai.chat.completions.create', input: request, output: answer },
      { spanId: 'second', kind: 'model', name: 'openai.chat.completions.create', input: request, output: answer },
    ] }, { mode: 'until_span', untilSpan: 'first' });
    await replay.run(async () => { await client.chat.completions.create(request); await client.chat.completions.create(request); });
    expect(create).toHaveBeenCalledTimes(2);
    expect(replay.getStatus()).toMatchObject({ hits: 0, reason: 'checkpoint', liveCalls: 2 });
  });

  it('rejects normalized or incomplete answers as model cache entries and keeps later calls live', async () => {
    const create = vi.fn(async (_request: unknown) => answer);
    const client = wrapOpenAI({ chat: { completions: { create } } });
    const replay = session('openai.chat.completions.create', request, { text: 'not a provider response' });
    await replay.run(async () => { await client.chat.completions.create(request); await client.chat.completions.create(request); });
    expect(create).toHaveBeenCalledTimes(2);
    expect(replay.getStatus()).toMatchObject({ hits: 0, liveCalls: 2, reason: 'missing_recording' });
  });

  it('keeps tools-only model calls live without breaking later tool replay', async () => {
    const create = vi.fn(async (_request: unknown) => answer);
    const client = wrapOpenAI({ chat: { completions: { create } } });
    const replay = session('openai.chat.completions.create', request, answer, 'tools_only');
    await replay.run(async () => {
      await client.chat.completions.create(request);
      expect(replay.decide('tool', 'lookup', {}).outcome).toBe('HIT');
    });
    expect(create).toHaveBeenCalledTimes(1);
    expect(replay.getStatus()).toMatchObject({ hits: 1, liveCalls: 1, switchedToLive: false });
  });

  it('restores a real Google response instance with working text and tool-call getters', async () => {
    const input = { model: 'gemini-test', contents: 'Hello' };
    const saved = { candidates: [{ content: { role: 'model', parts: [{ text: 'Saved' }, { functionCall: { name: 'lookup', args: { id: '1' } } }] }, finishReason: 'STOP' }],
      usageMetadata: { promptTokenCount: 12, candidatesTokenCount: 3, totalTokenCount: 15 } };
    const generateContent = vi.fn(async (_request: unknown): Promise<GenerateContentResponse> => { throw new Error('Must use saved response'); });
    const client = wrapGoogleGenAI({ models: { generateContent } });
    const replay = session('google_genai.models.generate_content', input, saved);
    const response = await replay.run(() => client.models.generateContent(input));
    expect(response).toBeInstanceOf(GenerateContentResponse);
    expect(response.text).toBe('Saved');
    expect(response.functionCalls).toEqual([{ name: 'lookup', args: { id: '1' } }]);
    expect(generateContent).not.toHaveBeenCalled();
    expect(only().attributes['neatlogs.llm.token_count.total']).toBe(0);
  });

  it('returns full Responses API output with its convenience text unchanged', async () => {
    const input = { model: 'test-model', input: 'Hello' };
    const saved = { id: 'response-saved', object: 'response', created_at: 123, model: 'test-model', status: 'completed',
      output: [{ type: 'message', role: 'assistant', content: [{ type: 'output_text', text: 'Saved', annotations: [] }] }], output_text: 'Saved' };
    const create = vi.fn(async (_request: unknown) => saved);
    const client = wrapOpenAI({ responses: { create } });
    const replay = session('openai.responses.create', input, saved);
    expect(await replay.run(() => client.responses.create(input))).toEqual(saved);
    expect(create).not.toHaveBeenCalled();
  });

  it('does not synthesize a stream or ignore caller transport options', async () => {
    const create = vi.fn(async (_request: unknown, _options?: unknown) => answer);
    const client = wrapOpenAI({ chat: { completions: { create } } });
    const replay = session('openai.chat.completions.create', request, answer);
    await replay.run(async () => { await client.chat.completions.create(request, { headers: { 'x-mode': 'changed' } }); });
    expect(create).toHaveBeenCalledTimes(1);
    expect(replay.getStatus()).toMatchObject({ hits: 0, reason: 'missing_recording' });
    const streaming = session('openai.chat.completions.create', { ...request, stream: true }, answer);
    await streaming.run(() => client.chat.completions.create({ ...request, stream: true }));
    expect(create).toHaveBeenCalledTimes(2);
    expect(streaming.getStatus().hits).toBe(0);
  });
});
