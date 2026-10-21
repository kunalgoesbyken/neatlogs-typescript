import { createHash, randomUUID } from 'node:crypto';
import { lstat, mkdir, open, readFile, rename, rm } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { z } from 'zod';
import { replaySettingsSchema, replayStatusSchema } from './contracts.js';
import { agentConfigSchema } from './local-command.js';
import { ALL_REPLAY_MAX_CASES, datasetSelectionDigest, selectedCaseSchema, type SelectedCase } from './all-selection.js';

export const allReplayConfigurationSchema = z.object({
  projectId: z.string().uuid(), datasetId: z.string().uuid(), versionId: z.string().uuid(), baseUrl: z.string().url(),
  cwd: z.string().min(1), agentFile: z.string().min(1), agentDigest: z.string().regex(/^[a-f0-9]{64}$/),
  label: z.string().min(1).max(120), trials: z.number().int().min(1).max(10), batchSize: z.number().int().min(1).max(100),
  deadlineSeconds: z.number().int().min(60).max(3600), replay: replaySettingsSchema,
  params: z.record(z.string().min(1).max(100), z.union([z.string().max(2000), z.number().finite(), z.boolean(), z.null()])).refine(value => Object.keys(value).length <= 20),
  grading: z.object({ formVersionId: z.string().uuid(), evaluatorIds: z.array(z.string().uuid()).min(1).max(20) }).strict().nullable(),
}).strict();
const progressSchema = z.discriminatedUnion('kind', [
  z.object({ kind: z.literal('pending') }).strict(),
  z.object({ kind: z.literal('executing') }).strict(),
  z.object({ kind: z.literal('upload_pending'), traceId: z.string().regex(/^[a-f0-9]{32}$/) }).strict(),
  z.object({ kind: z.literal('accepted'), traceId: z.string().regex(/^[a-f0-9]{32}$/), outputId: z.string().uuid() }).strict(),
  z.object({ kind: z.literal('local_complete') }).strict(),
  z.object({ kind: z.literal('failed'), message: z.string().max(2000) }).strict(),
]);
const executionSchema = z.object({
  sourceTraceId: z.string().min(1).max(200), runId: z.string().uuid().nullable(),
  status: replayStatusSchema.nullable(), exitCode: z.number().int().nullable(),
}).strict();
const slotSchema = z.object({ caseId: z.string().uuid(), trial: z.number().int().min(1).max(10), uploadKey: z.string().uuid(), progress: progressSchema, execution: executionSchema.nullable() }).strict();
const runSchema = z.object({ runId: z.string().uuid(), state: z.string(), deadlineAt: z.string().datetime(), itemCount: z.number().int().nonnegative() }).strict();
const batchSchema = z.object({ offset: z.number().int().nonnegative(), registrationKey: z.string().uuid(), run: runSchema.nullable(), slots: z.array(slotSchema).min(1).max(1000) }).strict();
const stateSchema = z.object({
  version: z.literal(1), groupId: z.string().uuid(), configuration: allReplayConfigurationSchema,
  configurationDigest: z.string().regex(/^[a-f0-9]{64}$/), selectionDigest: z.string().regex(/^[a-f0-9]{64}$/),
  cases: z.array(selectedCaseSchema).min(1).max(ALL_REPLAY_MAX_CASES), batches: z.array(batchSchema).min(1).max(ALL_REPLAY_MAX_CASES),
}).strict();
export type AllReplayConfiguration = z.infer<typeof allReplayConfigurationSchema>;
export type AllReplayState = z.infer<typeof stateSchema>;
export type AllReplaySlot = z.infer<typeof slotSchema>;
export const agentConfigDigest = (value: z.infer<typeof agentConfigSchema>) => createHash('sha256').update(JSON.stringify(agentConfigSchema.parse(value))).digest('hex');
export const configurationDigest = (value: AllReplayConfiguration) => createHash('sha256').update(JSON.stringify(allReplayConfigurationSchema.parse(value))).digest('hex');

export function createAllReplayState(configuration: AllReplayConfiguration, cases: SelectedCase[]): AllReplayState {
  const batches: AllReplayState['batches'] = [];
  for (let offset = 0; offset < cases.length; offset += configuration.batchSize) batches.push({
    offset, registrationKey: randomUUID(), run: null,
    slots: cases.slice(offset, offset + configuration.batchSize).flatMap(item => Array.from({ length: configuration.trials }, (_, trial) => ({
      caseId: item.caseId, trial: trial + 1, uploadKey: randomUUID(), progress: { kind: 'pending' as const }, execution: null,
    }))),
  });
  return stateSchema.parse({ version: 1, groupId: randomUUID(), configuration, configurationDigest: configurationDigest(configuration),
    selectionDigest: datasetSelectionDigest({ ...configuration, cases }), cases, batches });
}

export async function readAllReplayState(path: string): Promise<AllReplayState> {
  const info = await lstat(path);
  if (!info.isFile() || info.size > 16 * 1024 * 1024) throw new Error('The saved run state must be a regular file smaller than 16 MiB.');
  const state = stateSchema.parse(JSON.parse(await readFile(path, 'utf8'))), config = state.configuration;
  if (configurationDigest(config) !== state.configurationDigest || datasetSelectionDigest({ ...config, cases: state.cases }) !== state.selectionDigest ||
      state.cases.some((item, index) => item.position !== index) || new Set(state.cases.map(item => item.caseId)).size !== state.cases.length ||
      state.batches.length !== Math.ceil(state.cases.length / config.batchSize)) throw new Error('The saved selection or command settings were changed. Start a new state file.');
  for (const [index, batch] of state.batches.entries()) {
    const expected = state.cases.slice(index * config.batchSize, (index + 1) * config.batchSize);
    if (batch.offset !== index * config.batchSize || batch.slots.length !== expected.length * config.trials ||
        batch.slots.some((slot, offset) => slot.caseId !== expected[Math.floor(offset / config.trials)]?.caseId || slot.trial !== offset % config.trials + 1))
      throw new Error('The saved batch selection was changed. Start a new state file.');
    for (const slot of batch.slots) {
      const activity = slot.execution, report = activity?.status;
      if (slot.progress.kind === 'pending' && activity || slot.progress.kind !== 'pending' && !activity ||
          activity && activity.runId !== (batch.run?.runId ?? null) ||
          report && (report.sourceTraceId !== activity?.sourceTraceId || report.mode !== config.replay.mode || activity.exitCode === null) ||
          !report && activity?.exitCode !== null && activity?.exitCode !== undefined ||
          ['accepted', 'upload_pending', 'local_complete'].includes(slot.progress.kind) && (!report || activity?.exitCode !== 0) ||
          (slot.progress.kind === 'accepted' || slot.progress.kind === 'upload_pending') &&
            (!batch.run || report?.traceIds.length !== 1 || report.traceIds[0] !== slot.progress.traceId) ||
          slot.progress.kind === 'local_complete' && batch.run ||
          slot.progress.kind === 'executing' && report)
        throw new Error('Saved replay activity does not match its example, source or run.');
    }
  }
  return state;
}

export async function saveAllReplayState(path: string, state: AllReplayState, create = false) {
  const bytes = JSON.stringify(state);
  if (Buffer.byteLength(bytes) > 16 * 1024 * 1024) throw new Error('Saved run state exceeds 16 MiB.');
  const target = create ? path : `${path}.${randomUUID()}.tmp`;
  const file = await open(target, 'wx', 0o600);
  try { await file.writeFile(bytes); await file.sync(); } finally { await file.close(); }
  if (!create) {
    try { await rename(target, path); } finally { await rm(target, { force: true }); }
  }
  const directory = await open(dirname(path), 'r');
  try { await directory.sync(); } finally { await directory.close(); }
}

export function assertReplayReportCapacity(state: AllReplayState) {
  if (Buffer.byteLength(JSON.stringify(state)) + 16 * 1024 > 16 * 1024 * 1024)
    throw new Error('Saved run state has no room for another replay report. The next agent command was not executed.');
}

/** One process owns the state; a crashed command is never silently replayed. */
export async function withAllReplayLock<T>(rawPath: string, execute: (path: string) => Promise<T>): Promise<T> {
  const path = resolve(rawPath), lockPath = `${path}.lock`;
  await mkdir(dirname(path), { recursive: true, mode: 0o700 });
  let lock;
  try { lock = await open(lockPath, 'wx', 0o600); }
  catch (error) {
    if (error instanceof Error && 'code' in error && error.code === 'EEXIST') throw new Error(`This saved run is locked: ${lockPath}. If its process stopped, inspect its outcome before removing the lock and continuing.`);
    throw error;
  }
  try { await lock.writeFile(JSON.stringify({ pid: process.pid })); return await execute(path); }
  finally { await lock.close(); await rm(lockPath, { force: true }); }
}
