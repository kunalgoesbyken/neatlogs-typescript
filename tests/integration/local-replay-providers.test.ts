import { describe, expect, it, vi } from 'vitest';
import { GenerateContentResponse } from '@google/genai';
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
