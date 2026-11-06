import { describe, expect, it } from 'vitest';
import { span } from '../../src/decorators/index.js';
import { ReplaySession, getReplayStatus, replayModel } from '../../src/replay/session.js';
import { replayRecordingFromCase } from '../../src/replay/frozen-case.js';
import type { ReplayEntry } from '../../src/replay/contracts.js';

const tool = (spanId: string, output: unknown = { balance: 40 }): ReplayEntry => ({
  spanId, kind: 'tool', name: 'lookup', input: { customer: '1', region: 'EU' }, output: output as ReplayEntry['output'],
});
const recording = (entries: ReplayEntry[]) => ({ version: 1, sourceTraceId: 'trace-1', entries });

describe('local code replay', () => {
  it('runs real orchestration, serves copied tool results, and keeps model calls live in tools_only', async () => {
    let calls = 0, modelCalls = 0;
    const lookup = span({ kind: 'TOOL', toolName: 'lookup', replay: true }, async (_input: { customer: string; region: string }) => { calls++; return { balance: 0 }; });
    const saved = recording([tool('s1'), tool('s2')]);
    const session = new ReplaySession(saved, { mode: 'tools_only' });
    saved.entries[0]!.output = { balance: 99 };
    const result = await session.run(async () => {
      const first = await lookup({ region: 'EU', customer: '1' });
      first.balance = 100;
      const second = await lookup({ customer: '1', region: 'EU' });
      return replayModel('model', {}, () => 'cached', () => { modelCalls++; return second.balance > 30 ? 'approve' : 'reject'; });
    });
    expect(result).toBe('approve'); expect(calls).toBe(0); expect(modelCalls).toBe(1);
    expect(session.getStatus()).toMatchObject({ hits: 2, liveCalls: 1, switchedToLive: false });
    expect(getReplayStatus()).toBeNull();
  });

  it('switches permanently to real code at the first miss', () => {
    let live = 0;
    const lookup = span({ kind: 'TOOL', name: 'lookup', replay: true }, (input: unknown) => { live++; return input; });
    const session = new ReplaySession(recording([tool('s1')]), { mode: 'tools_only' });
    session.run(() => { lookup({ customer: 'different' }); lookup({ customer: '1', region: 'EU' }); });
    expect(live).toBe(2);
    expect(session.getStatus()).toMatchObject({ hits: 0, liveCalls: 2, reason: 'missing_recording', firstMiss: { kind: 'tool', name: 'lookup' } });
  });

  it('replays the prefix, runs the checkpoint live, and never serves later cached results', () => {
    const entries = [tool('s1'), { spanId: 'm1', kind: 'model' as const, name: 'classify', input: ['hello'], output: 'old' }, tool('s2')];
    const session = new ReplaySession(recording(entries), { mode: 'until_span', untilSpan: 'm1' });
    session.run(() => {
      expect(session.decide('tool', 'lookup', { customer: '1', region: 'EU' }).outcome).toBe('HIT');
      expect(replayModel('classify', ['hello'], value => String(value), () => 'new')).toBe('new');
      expect(session.decide('tool', 'lookup', { customer: '1', region: 'EU' }).outcome).toBe('LIVE');
    });
    expect(session.getStatus()).toMatchObject({ reason: 'checkpoint', hits: 1, liveCalls: 2 });
  });

  it('restores model results through an explicit decoder and treats out-of-order later work as a miss', () => {
    const entries: ReplayEntry[] = [{ spanId: 'm0', kind: 'model', name: 'classify', input: 'hello', output: { text: 'saved' } }, tool('s1'), { ...tool('s2'), name: 'later' }];
    const session = new ReplaySession(recording(entries), { mode: 'until_span', untilSpan: 's1' });
    session.run(() => {
      expect(replayModel('classify', 'hello', value => ({ text: (value as { text: string }).text }), () => ({ text: 'live' }))).toEqual({ text: 'saved' });
      expect(session.decide('tool', 'later', { customer: '1', region: 'EU' }).outcome).toBe('MISS');
    });
  });

  it('preserves sync values and explicit promise factories and consumes repeated results in order', async () => {
    const sync = span({ kind: 'TOOL', name: 'lookup', replay: true }, (_: unknown) => 9);
    const promise = span({ kind: 'TOOL', name: 'lookup', replay: { result: 'promise' } }, (_: unknown) => Promise.resolve(9));
    const session = new ReplaySession(recording([tool('s1', 1), tool('s2', 2)]), { mode: 'tools_only' });
    await session.run(async () => {
      expect(sync({ customer: '1', region: 'EU' })).toBe(1);
      const result = promise({ customer: '1', region: 'EU' });
      expect(result).toBeInstanceOf(Promise); expect(await result).toBe(2);
    });
  });

  it('isolates concurrent sessions and rejects reuse, nesting, duplicate identities and missing checkpoints', async () => {
    const first = new ReplaySession(recording([tool('s1', 'first')]), { mode: 'tools_only' });
    const second = new ReplaySession(recording([tool('s1', 'second')]), { mode: 'tools_only' });
    const execute = span({ kind: 'TOOL', name: 'lookup', replay: true }, async (_: unknown) => 'live');
    const run = (session: ReplaySession) => session.run(async () => { await Promise.resolve(); return execute({ customer: '1', region: 'EU' }); });
    expect(await Promise.all([run(first), run(second)])).toEqual(['first', 'second']);
    expect(() => first.run(() => null)).toThrow('new ReplaySession');
    expect(() => new ReplaySession(recording([tool('s1'), tool('s1')]), { mode: 'tools_only' })).toThrow('unique');
    expect(() => new ReplaySession(recording([]), { mode: 'until_span', untilSpan: 'no' })).toThrow('checkpoint');
    expect(() => new ReplaySession(recording([]), { mode: 'tools_only' }).run(() => new ReplaySession(recording([]), { mode: 'tools_only' }).run(() => null))).toThrow('Nested');
  });

  it('reads complete frozen tools and rejects errors, missing output, redaction and ambiguous identities', () => {
    const saved = { span_id: 't1', span_type: 'TOOL', semantic: { kind: 'TOOL', definition: { name: 'lookup' }, call: { id: 'c1', name: 'lookup', arguments: { customer: '1' } }, result: { call_id: 'c1', is_error: false, value: null } } };
    const make = (spans: unknown[]) => replayRecordingFromCase({ source: { traceId: 'trace-1' }, context: { spans } });
    expect(make([saved]).recording.entries[0]?.output).toBeNull();
    expect(make([saved, saved]).recording.entries).toEqual([]);
    expect(make([{ ...saved, status_code: 2 }]).skippedTools).toBe(1);
    expect(make([{ ...saved, attributes: { 'pii.server_redacted': true } }]).skippedTools).toBe(1);
    expect(make([{ ...saved, semantic: { ...saved.semantic, result: { call_id: 'c1', is_error: false } } }]).skippedTools).toBe(1);
    expect(make([{ ...saved, semantic: { ...saved.semantic, result: { call_id: 'other', is_error: false, value: 'wrong' } } }]).skippedTools).toBe(1);
  });
});
