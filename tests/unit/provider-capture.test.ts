import { describe, expect, it, vi } from 'vitest';
import { getProviderTracer } from '../../src/core/auto-root.js';
import { applyMask, registerMask } from '../../src/core/mask.js';
import { MODEL_CAPTURE_ATTRIBUTE, prepareProviderCapture, readProviderCapture } from '../../src/replay/provider-capture.js';
import { replayRecordingFromCase } from '../../src/replay/frozen-case.js';
import { useSpans } from './failed-calls/harness.js';

const { only } = useSpans();
const input = { model: 'local-model', messages: [{ role: 'user', content: 'Question' }] };
const output = { id: 'saved', object: 'chat.completion', created: 1, model: 'local-model',
  choices: [{ index: 0, finish_reason: 'stop', message: { role: 'assistant', content: 'Answer' } }] };
function capture(request: unknown = input, response: unknown = output, supported = true) {
  const span = getProviderTracer('capture-test').startSpan('provider');
  prepareProviderCapture(span, 'openai.chat.completions.create', request, supported)(response);
  span.end();
  return only().attributes[MODEL_CAPTURE_ATTRIBUTE];
}
function imported(attributes: Record<string, unknown>, extra: Record<string, unknown> = {}) {
  return replayRecordingFromCase({ source: { traceId: 'saved' }, context: { spans: [{
    span_id: 'model', span_type: 'LLM', attributes, ...extra,
  }] } }).recording.entries;
}

describe('raw provider capture eligibility', () => {
  it('freezes actual request data before the provider can mutate it', () => {
    const mutable = structuredClone(input);
    const span = getProviderTracer('capture-test').startSpan('provider');
    const finish = prepareProviderCapture(span, 'openai.chat.completions.create', mutable, true);
    mutable.messages[0].content = 'Changed';
    finish(output); span.end();
    expect(readProviderCapture(only().attributes[MODEL_CAPTURE_ATTRIBUTE]).input).toEqual(input);
  });

  it.each([
    ['credentials', { ...input, apiKey: 'must-not-be-saved' }],
    ['HTTP options', { ...input, config: { httpOptions: { headers: { authorization: 'must-not-be-saved' } } } }],
    ['inline media', { ...input, messages: [{ role: 'user', content: [{ type: 'image_url', image_url: { url: 'data:image/png;base64,aGVsbG8=' } }] }] }],
    ['media URL', { ...input, messages: [{ role: 'user', content: [{ type: 'image_url', image_url: { url: 'https://example.test/image.png?token=private' } }] }] }],
    ['oversized content', { ...input, messages: [{ role: 'user', content: 'x'.repeat(1024 * 1024) }] }],
    ['non-JSON data', { ...input, state: new Date() }],
  ])('omits %s instead of reconstructing a different request', (_label, request) => {
    expect(capture(request)).toBeUndefined();
  });

  it('does not invoke getters or toJSON while inspecting evidence', () => {
    const getter = vi.fn(() => 'hidden');
    const request = { ...input, toJSON: vi.fn(() => input) };
    Object.defineProperty(request, 'computed', { enumerable: true, get: getter });
    expect(capture(request)).toBeUndefined();
    expect(getter).not.toHaveBeenCalled();
    expect(request.toJSON).not.toHaveBeenCalled();
  });

  it('does not invoke array getters', () => {
    const getter = vi.fn(() => 'hidden');
    const messages = ['placeholder'];
    Object.defineProperty(messages, '0', { enumerable: true, get: getter });
    expect(capture({ ...input, messages })).toBeUndefined();
    expect(getter).not.toHaveBeenCalled();
  });

  it.each([false, true])('unsupported calls or incomplete output stay uncached (%s)', supported => {
    expect(capture(input, supported ? { text: 'normalized only' } : output, supported)).toBeUndefined();
  });

  it('rejects redacted or clipped evidence instead of treating normalized text as a response', () => {
    const raw = capture();
    expect(imported({ [MODEL_CAPTURE_ATTRIBUTE]: raw })).toHaveLength(1);
    expect(imported({ 'output.value': JSON.stringify(output) })).toEqual([]);
    expect(imported({ [MODEL_CAPTURE_ATTRIBUTE]: String(raw).replace('Answer', '[redacted]') })).toEqual([]);
    expect(imported({ [MODEL_CAPTURE_ATTRIBUTE]: String(raw).slice(0, -10) })).toEqual([]);
    for (const flag of ['pii.server_redacted', 'neatlogs.stream.incomplete', 'neatlogs.payload.truncated'])
      expect(imported({ [MODEL_CAPTURE_ATTRIBUTE]: raw, [flag]: true })).toEqual([]);
    expect(imported({ [MODEL_CAPTURE_ATTRIBUTE]: raw, 'neatlogs.capture_fidelity': 'truncated' })).toEqual([]);
    expect(imported({ [MODEL_CAPTURE_ATTRIBUTE]: raw }, { output_content_state: 'redacted' })).toEqual([]);
  });

  it.each(['global', 'per-span'])('drops raw capture before a %s mask that only knows normalized I/O', async mode => {
    const raw = capture();
    const mask = vi.fn((data: Record<string, unknown>) => {
      expect(data.attributes).not.toHaveProperty(MODEL_CAPTURE_ATTRIBUTE);
      return { ...data, output: '[redacted]' };
    });
    const attributes = { [MODEL_CAPTURE_ATTRIBUTE]: raw, ...(mode === 'per-span' ? { 'neatlogs.mask_id': registerMask(mask) } : {}) };
    const result = await applyMask({ attributes }, mode === 'global' ? mask : null);
    expect(mask).toHaveBeenCalledOnce();
    expect(result?.attributes).not.toHaveProperty(MODEL_CAPTURE_ATTRIBUTE);
  });

  it('preserves complete evidence when no custom mask is configured', async () => {
    const raw = capture();
    const result = await applyMask({ attributes: { [MODEL_CAPTURE_ATTRIBUTE]: raw } }, null);
    expect(result?.attributes[MODEL_CAPTURE_ATTRIBUTE]).toBe(raw);
  });
});
