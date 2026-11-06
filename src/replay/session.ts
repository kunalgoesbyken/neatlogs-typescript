import { AsyncLocalStorage } from 'node:async_hooks';
import type { Span } from '@opentelemetry/api';
import { getActiveNeatlogsSpan } from '../core/provider.js';
import { parseReplayRecording, replayKey, replayRunBindingSchema, type ReplayEntry, type ReplayOptions, type ReplayRecording, type ReplayStatus } from './contracts.js';

type Decision = { outcome: 'HIT'; entry: ReplayEntry } | { outcome: 'MISS' | 'LIVE' };
type Reservation = { canRestore: boolean; consume: (restore: boolean) => Decision };
const symbol = Symbol.for('neatlogs.local_replay');
type ReplayGlobal = typeof globalThis & { [symbol]?: AsyncLocalStorage<ReplaySession> };
const global = globalThis as ReplayGlobal;
const storage = global[symbol] ?? (global[symbol] = new AsyncLocalStorage<ReplaySession>());
const environmentSymbol = Symbol.for('neatlogs.local_replay_environment');
type EnvironmentGlobal = typeof globalThis & { [environmentSymbol]?: ReplaySession };
const environment = globalThis as EnvironmentGlobal;
const current = () => storage.getStore() ?? environment[environmentSymbol];

export function setEnvironmentReplay(session: ReplaySession): void {
  if (environment[environmentSymbol]) throw new Error('Environment replay is already configured.');
  environment[environmentSymbol] = session;
}

export class ReplaySession {
  private readonly recording: ReplayRecording;
  private readonly entries = new Map<string, ReplayEntry[]>();
  private readonly positions = new Map<string, number>();
  private readonly options: ReplayOptions;
  private state: ReplayStatus;
  private used = false;
  private sequence = 0;
  private liveAfter = Infinity;
  private lastConsumedHit = -1;
  private readonly checkpointIndex: number;

  constructor(recording: unknown, options: ReplayOptions) {
    this.recording = parseReplayRecording(recording);
    if (options.mode !== 'tools_only' && options.mode !== 'until_span') throw new Error('Unsupported replay mode.');
    if (options.mode === 'until_span' && (!options.untilSpan || !this.recording.entries.some(entry => entry.spanId === options.untilSpan)))
      throw new Error('The replay checkpoint must exist in this recording.');
    if (options.mode === 'tools_only' && options.untilSpan) throw new Error('tools_only does not accept a checkpoint.');
    this.options = { ...options, ...(options.run ? { run: replayRunBindingSchema.parse(options.run) } : {}) };
    this.checkpointIndex = this.recording.entries.findIndex(entry => entry.spanId === options.untilSpan);
    this.state = { sourceTraceId: this.recording.sourceTraceId, mode: options.mode, hits: 0, liveCalls: 0, switchedToLive: false, reason: null, firstMiss: null, traceIds: [] };
    for (const entry of this.recording.entries) {
      const key = replayKey(entry.kind, entry.name, entry.input);
      const sequence = this.entries.get(key) ?? [];
      sequence.push(entry);
      this.entries.set(key, sequence);
    }
  }

  getStatus(): ReplayStatus { return structuredClone(this.state); }

  stamp(span: Span): void {
    const traceId = span.spanContext().traceId;
    // Two identities are enough to report ambiguity; a run slot accepts one trace.
    if (/^[a-f0-9]{32}$/.test(traceId) && !/^0+$/.test(traceId) && !this.state.traceIds.includes(traceId) && this.state.traceIds.length < 2)
      this.state.traceIds.push(traceId);
    span.setAttribute('neatlogs.origin', 'simulation');
    span.setAttribute('neatlogs.replay.source_trace_id', this.state.sourceTraceId);
    span.setAttribute('neatlogs.replay.mode', this.state.mode);
    if (this.options.run) {
      span.setAttribute('neatlogs.replay.run_id', this.options.run.runId);
      span.setAttribute('neatlogs.replay.case_id', this.options.run.caseId);
      span.setAttribute('neatlogs.replay.trial', this.options.run.trial);
    }
  }

  decide(kind: ReplayEntry['kind'], name: string, input: unknown, canRestore?: (output: unknown) => boolean): Decision {
    const sequence = this.sequence++;
    if (this.state.switchedToLive || this.options.mode === 'tools_only' && kind === 'model') {
      this.state.liveCalls++;
      return { outcome: 'LIVE' };
    }
    let key: string;
    try { key = replayKey(kind, name, input); }
    catch { return this.miss(kind, name); }
    const position = this.positions.get(key) ?? 0;
    const entry = this.entries.get(key)?.[position];
    if (!entry) return this.miss(kind, name);
    if (entry.spanId === this.options.untilSpan) {
      this.state.switchedToLive = true;
      this.state.reason = 'checkpoint';
      this.liveAfter = Math.min(this.liveAfter, sequence);
      this.state.liveCalls++;
      return { outcome: 'LIVE' };
    }
    // Never replay an entry recorded after the requested checkpoint, even if
    // changed orchestration calls it before reaching that checkpoint locally.
    if (this.options.untilSpan && this.recording.entries.indexOf(entry) > this.checkpointIndex)
      return this.miss(kind, name);
    if (canRestore && !canRestore(entry.output)) return this.miss(kind, name);
    this.positions.set(key, position + 1);
    this.state.hits++;
    this.lastConsumedHit = Math.max(this.lastConsumedHit, sequence);
    return { outcome: 'HIT', entry: structuredClone(entry) };
  }

  /** Reserve repeated inputs in invocation order; count only consumed results. */
  reserve(kind: ReplayEntry['kind'], name: string, input: unknown, canRestore?: (output: unknown) => boolean): Reservation {
    const sequence = this.sequence, lastConsumedHit = this.lastConsumedHit;
    const selected = this.decide(kind, name, input, canRestore);
    this.lastConsumedHit = lastConsumedHit;
    if (selected.outcome === 'HIT') this.state.hits--;
    else this.state.liveCalls--;
    let consumed: Decision | undefined;
    return { canRestore: selected.outcome === 'HIT', consume: restore => {
      if (consumed) return consumed;
      if (selected.outcome === 'HIT') {
        if (!restore && this.lastConsumedHit > sequence) throw new Error('Request raw-response helpers before consuming a later replayed response; HTTP metadata was not recorded.');
        if (!restore) return consumed = this.miss(kind, name, sequence);
        if (sequence > this.liveAfter) { this.state.liveCalls++; return consumed = { outcome: 'LIVE' }; }
        this.state.hits++;
        this.lastConsumedHit = Math.max(this.lastConsumedHit, sequence);
      } else this.state.liveCalls++;
      return consumed = selected;
    } };
  }

  private miss(kind: ReplayEntry['kind'], name: string, sequence = this.sequence - 1): Decision {
    this.state.switchedToLive = true;
    if (sequence <= this.liveAfter) {
      this.liveAfter = sequence;
      this.state.reason = 'missing_recording';
      this.state.firstMiss = { kind, name };
    }
    this.state.liveCalls++;
    return { outcome: 'MISS' };
  }

  run<T>(execute: () => T): T {
    if (current()) throw new Error('Nested local replay sessions are not supported.');
    if (this.used) throw new Error('Create a new ReplaySession for each local run.');
    this.used = true;
    return storage.run(this, execute);
  }
}

export function getReplayStatus(): ReplayStatus | null { return current()?.getStatus() ?? null; }
export function hasReplaySession(): boolean { return current() !== undefined; }

export function stampReplaySpan(span: Span): void {
  current()?.stamp(span);
}

export function lookupReplay(kind: ReplayEntry['kind'], name: string, input: unknown, canRestore?: (output: unknown) => boolean): Decision | null {
  const session = current();
  if (!session) return null;
  const decision = session.decide(kind, name, input, canRestore);
  const span = getActiveNeatlogsSpan();
  if (span) stampReplaySpan(span);
  span?.setAttribute('neatlogs.replay.outcome', decision.outcome);
  if (decision.outcome === 'HIT') span?.setAttribute('neatlogs.replay.source_span_id', decision.entry.spanId);
  return decision;
}

export function reserveReplay(kind: ReplayEntry['kind'], name: string, input: unknown, canRestore?: (output: unknown) => boolean): Reservation | null {
  const session = current();
  if (!session) return null;
  const reservation = session.reserve(kind, name, input, canRestore), span = getActiveNeatlogsSpan();
  return { canRestore: reservation.canRestore, consume: restore => {
    const decision = reservation.consume(restore);
    if (span) session.stamp(span);
    span?.setAttribute('neatlogs.replay.outcome', decision.outcome);
    if (decision.outcome === 'HIT') span?.setAttribute('neatlogs.replay.source_span_id', decision.entry.spanId);
    return decision;
  } };
}

/** Callers supply a decoder so cached JSON cannot masquerade as an SDK class. */
export function replayModel<T>(name: string, input: unknown, decode: (saved: unknown) => T, live: () => T): T {
  const decision = lookupReplay('model', name, input);
  return decision?.outcome === 'HIT' ? decode(decision.entry.output) : live();
}
