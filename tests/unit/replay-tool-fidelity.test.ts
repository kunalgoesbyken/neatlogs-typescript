import { describe, expect, it, vi } from 'vitest';
import { span } from '../../src/decorators/index.js';
import { applyMask } from '../../src/core/mask.js';
import { ReplaySession } from '../../src/replay/session.js';
import { replayRecordingFromCase } from '../../src/replay/frozen-case.js';
import { useSpans } from './failed-calls/harness.js';

const { only } = useSpans();
function saved(attributes: Record<string, unknown>, semantic?: unknown) {
  return { source: { traceId: 'source' }, context: { spans: [{ span_id: 'tool', span_type: 'TOOL', span_name: 'lookup', tool_name: '', attributes,
    input_value: attributes['input.value'], output_value: attributes['output.value'], ...(semantic ? { semantic } : {}),
  }] } };
}
function semantic(output: unknown) {
  return { kind: 'TOOL', definition: { name: 'lookup' }, call: { id: 'call', name: 'lookup', arguments: { id: 1 } }, result: { call_id: 'call', is_error: false, value: output } };
}

describe('frozen tool execution fidelity', () => {
  it('captures the resolved name and replays an actual recorded tool without an explicit toolName', async () => {
    const live = vi.fn((value: unknown) => ({ answer: value }));
    const lookup = span({ kind: 'TOOL', name: 'lookup', replay: true }, live);
    const output = await lookup({ id: 1 });
    const attributes = only().attributes;
    expect(attributes['tool.name']).toBe('lookup');
    const imported = replayRecordingFromCase(saved(attributes));
    expect(imported.skippedTools).toBe(0);
    expect(imported.recording.entries[0]?.name).toBe('lookup');
    const replay = new ReplaySession(imported.recording, { mode: 'tools_only' });
    expect(await replay.run(() => lookup({ id: 1 }))).toEqual(output);
    expect(live).toHaveBeenCalledOnce();
    expect(replay.getStatus().hits).toBe(1);
  });

  it.each(['raw', 'semantic'])('rejects media placeholders through the %s saved path', async path => {
    const original = { image: 'data:image/png;base64,' + Buffer.alloc(100_001).toString('base64') };
    const lookup = span({ kind: 'TOOL', name: 'lookup' }, () => original);
    expect(await lookup({ id: 1 })).toBe(original);
    const attributes = only().attributes;
    const captured = JSON.parse(String(attributes['output.value']));
    expect(captured.image).not.toBe(original.image);
    const imported = replayRecordingFromCase(saved(attributes, path === 'semantic' ? semantic(captured) : undefined));
    expect(imported.recording.entries).toEqual([]);
    expect(imported.skippedTools).toBe(1);
  });

  it.each(['raw', 'semantic'])('marks structural clipping at capture instead of replaying a changed deep output through %s', async path => {
    let original: unknown = 'value';
    for (let depth = 0; depth < 35; depth++) original = { nested: original };
    const live = vi.fn(() => original);
    const lookup = span({ kind: 'TOOL', name: 'lookup', replay: true }, live);
    expect(await lookup({ id: 1 })).toBe(original);
    const attributes = only().attributes;
    expect(attributes['neatlogs.capture_fidelity']).toBe('truncated');
    const captured = JSON.parse(String(attributes['output.value']));
    const imported = replayRecordingFromCase(saved(attributes, path === 'semantic' ? semantic(captured) : undefined));
    expect(imported.recording.entries).toEqual([]);
    const replay = new ReplaySession(imported.recording, { mode: 'tools_only' });
    expect(await replay.run(() => lookup({ id: 1 }))).toBe(original);
    expect(live).toHaveBeenCalledTimes(2);
    expect(replay.getStatus()).toMatchObject({ hits: 0, reason: 'missing_recording' });
  });

  it('does not mistake legitimate user text for an SDK structural marker', async () => {
    const lookup = span({ kind: 'TOOL', name: 'lookup' }, () => '[REDACTED_MEDIA_STRUCTURE]');
    await lookup({ id: 1 });
    const imported = replayRecordingFromCase(saved(only().attributes));
    expect(imported.recording.entries[0]?.output).toBe('[REDACTED_MEDIA_STRUCTURE]');
  });

  it.each(['hole-extra', 'symbol', 'hidden', 'accessor'])('never replays an array with lossy %s properties as plain JSON', async defect => {
    const output = defect === 'hole-extra' ? new Array(1) : ['original'];
    if (defect === 'hole-extra') Object.defineProperty(output, 'extra', { value: 'lost', enumerable: true });
    if (defect === 'symbol') Object.defineProperty(output, Symbol('extra'), { value: 'lost' });
    if (defect === 'hidden') Object.defineProperty(output, 'extra', { value: 'lost' });
    if (defect === 'accessor') Object.defineProperty(output, '0', { get: () => 'computed', enumerable: true });
    const live = vi.fn(() => output);
    const lookup = span({ kind: 'TOOL', name: 'lookup', replay: true }, live);
    expect(await lookup({ id: 1 })).toBe(output);
    const attributes = only().attributes;
    expect(attributes['neatlogs.capture_fidelity']).toBe('unavailable');
    const imported = replayRecordingFromCase(saved(attributes));
    expect(imported.recording.entries).toEqual([]);
    const replay = new ReplaySession(imported.recording, { mode: 'tools_only' });
    expect(await replay.run(() => lookup({ id: 1 }))).toBe(output);
    expect(live).toHaveBeenCalledTimes(2);
    expect(replay.getStatus().hits).toBe(0);
  });

  it('marks non-JSON circular tool output before serialization changes it', async () => {
    const output: Record<string, unknown> = {}; output.self = output;
    const lookup = span({ kind: 'TOOL', name: 'lookup' }, () => output);
    expect(await lookup({ id: 1 })).toBe(output);
    expect(only().attributes['neatlogs.capture_fidelity']).toBe('unavailable');
    expect(replayRecordingFromCase(saved(only().attributes)).recording.entries).toEqual([]);
  });

  it.each(['raw', 'semantic'])('rejects changed tool output from a custom mask through the %s saved path', async path => {
    const lookup = span({ kind: 'TOOL', name: 'lookup' }, () => ({ answer: 'original' }));
    await lookup({ id: 1 });
    const result = await applyMask({ attributes: { ...only().attributes } }, data => ({ ...data, attributes: { ...data.attributes, 'output.value': JSON.stringify({ answer: 'masked' }) } }));
    expect(result?.attributes['neatlogs.capture_fidelity']).toBe('masked');
    const imported = replayRecordingFromCase(saved(result!.attributes, path === 'semantic' ? semantic({ answer: 'masked' }) : undefined));
    expect(imported.recording.entries).toEqual([]);
    expect(imported.skippedTools).toBe(1);
  });
});
