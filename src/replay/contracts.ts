import { createHash } from 'node:crypto';
import { z } from 'zod';
import { json, type DatasetJson } from '../dataset/api.js';

const entry = z.object({
  spanId: z.string().min(1).max(200),
  kind: z.enum(['tool', 'model']),
  name: z.string().min(1).max(200),
  input: z.json(), output: z.json(),
}).strict();
export const replayRecordingSchema = z.object({
  version: z.literal(1),
  sourceTraceId: z.string().min(1).max(200),
  entries: z.array(entry).max(5000),
}).strict().superRefine((value, ctx) => {
  if (new Set(value.entries.map(item => item.spanId)).size !== value.entries.length)
    ctx.addIssue({ code: 'custom', message: 'Replay span identities must be unique.' });
});
export type ReplayRecording = z.infer<typeof replayRecordingSchema>;
export type ReplayEntry = ReplayRecording['entries'][number];
export const replayRunBindingSchema = z.object({
  runId: z.string().uuid(), caseId: z.string().uuid(), trial: z.number().int().min(1).max(10),
}).strict();
export type ReplayRunBinding = z.infer<typeof replayRunBindingSchema>;
export const replaySettingsSchema = z.discriminatedUnion('mode', [
  z.object({ mode: z.literal('tools_only') }).strict(),
  z.object({ mode: z.literal('until_span'), untilSpan: z.string().min(1).max(200) }).strict(),
]);
export type ReplaySettings = z.infer<typeof replaySettingsSchema>;
export type ReplayOptions = { mode: 'tools_only' | 'until_span'; untilSpan?: string; run?: ReplayRunBinding };
export type ReplayStatus = {
  sourceTraceId: string; mode: ReplayOptions['mode']; hits: number; liveCalls: number;
  switchedToLive: boolean; reason: 'checkpoint' | 'missing_recording' | null;
  firstMiss: { kind: ReplayEntry['kind']; name: string } | null;
  traceIds: string[];
};
export const replayStatusSchema = z.object({
  sourceTraceId: z.string().min(1).max(200), mode: z.enum(['tools_only', 'until_span']),
  hits: z.number().int().nonnegative(), liveCalls: z.number().int().nonnegative(),
  switchedToLive: z.boolean(), reason: z.enum(['checkpoint', 'missing_recording']).nullable(),
  firstMiss: z.object({ kind: z.enum(['tool', 'model']), name: z.string().max(200) }).strict().nullable(),
  traceIds: z.array(z.string().regex(/^[a-f0-9]{32}$/)).max(2),
}).strict();

function canonical(value: DatasetJson): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value !== null && typeof value === 'object')
    return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${canonical(value[key]!)}`).join(',')}}`;
  return JSON.stringify(value);
}
export function replayKey(kind: ReplayEntry['kind'], name: string, input: unknown): string {
  const encoded = canonical(json(input));
  if (Buffer.byteLength(encoded) > 1024 * 1024) throw new Error('Replay input exceeds 1 MiB.');
  return `${kind}:${name}:${createHash('sha256').update(encoded).digest('hex')}`;
}
export function parseReplayRecording(value: unknown): ReplayRecording {
  const encoded = JSON.stringify(json(value));
  if (Buffer.byteLength(encoded) > 16 * 1024 * 1024) throw new Error('Replay recording exceeds 16 MiB.');
  return replayRecordingSchema.parse(JSON.parse(encoded));
}
