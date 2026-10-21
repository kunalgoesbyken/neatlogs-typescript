import { readFile, stat } from 'node:fs/promises';
import { resolve } from 'node:path';
import { z } from 'zod';
import { DatasetApi, uuid, type DatasetApiOptions } from '../dataset/api.js';
import { enumerateSavedDataset } from './all-selection.js';
import { agentConfigDigest, allReplayConfigurationSchema, assertReplayReportCapacity, createAllReplayState, readAllReplayState, saveAllReplayState, withAllReplayLock, type AllReplayConfiguration, type AllReplaySlot, type AllReplayState } from './all-state.js';
import { replayRecordingFromCase } from './frozen-case.js';
import { parseAgentConfig, runLocalCommand } from './local-command.js';
import { LocalRunApi } from './run-api.js';
import { ReplaySession } from './session.js';

const sourceSchema = z.object({
  case: z.object({ caseId: z.string().uuid(), revisionId: z.string().uuid() }).passthrough(),
  position: z.number().int().nonnegative(),
});
const message = (error: unknown) => (error instanceof Error ? error.message : 'Local replay failed.').slice(0, 2000);
type Runner = { state: AllReplayState; path: string; dataset: DatasetApi; api: LocalRunApi | null; agent: ReturnType<typeof parseAgentConfig>; json: boolean };
type Batch = AllReplayState['batches'][number];

async function uploadPending(owner: Runner, batch: Batch, slot: AllReplaySlot) {
  if (slot.progress.kind !== 'upload_pending') return;
  if (!owner.api || !batch.run) throw new Error('The saved output has no registered run.');
  const traceId = slot.progress.traceId;
  const receipt = await owner.api.upload(batch.run.runId, { caseId: slot.caseId, trial: slot.trial, traceId }, { key: slot.uploadKey,
    waitMillis: Math.max(1, Math.min(60_000, Date.parse(batch.run.deadlineAt) - Date.now())) });
  slot.progress = { kind: 'accepted', traceId, outputId: receipt.outputId };
  await saveAllReplayState(owner.path, owner.state);
}

async function executeSlot(owner: Runner, batch: Batch, slot: AllReplaySlot) {
  if (slot.progress.kind === 'executing') throw new Error('A previous agent command has an unknown outcome. It will not run again automatically. Inspect it before starting another run.');
  if (slot.progress.kind === 'upload_pending') return uploadPending(owner, batch, slot);
  if (slot.progress.kind !== 'pending') return;
  const config = owner.state.configuration;
  const selected = owner.state.cases.find(item => item.caseId === slot.caseId);
  if (!selected) throw new Error('Saved example is outside the selection.');
  let remaining = batch.run ? Date.parse(batch.run.deadlineAt) - Date.now() : Infinity;
  if (remaining < 1000) throw new Error('This batch has expired. Remaining examples were not executed.');
  const source = sourceSchema.parse(await owner.dataset.request('GET', `/datasets/${config.datasetId}/cases/${slot.caseId}?versionId=${config.versionId}`));
  if (source.case.caseId !== selected.caseId || source.case.revisionId !== selected.revisionId || source.position !== selected.position)
    throw new Error('The saved example does not match the pinned selection.');
  const converted = replayRecordingFromCase(source.case);
  new ReplaySession(converted.recording, config.replay);
  if (owner.api && batch.run) {
    const status = await owner.api.status(batch.run.runId);
    if (!['queued', 'running'].includes(status.state)) throw new Error('This batch no longer accepts new outputs. The agent was not executed.');
  }
  remaining = batch.run ? Date.parse(batch.run.deadlineAt) - Date.now() : Infinity;
  if (remaining < 1000) throw new Error('This batch expired while loading its saved example. The agent was not executed.');
  assertReplayReportCapacity(owner.state);
  slot.execution = { sourceTraceId: converted.recording.sourceTraceId, runId: batch.run?.runId ?? null, status: null, exitCode: null };
  slot.progress = { kind: 'executing' };
  await saveAllReplayState(owner.path, owner.state);
  let result: Awaited<ReturnType<typeof runLocalCommand>>;
  try {
    result = await runLocalCommand({ recording: converted.recording, input: source.case.input,
      inputContext: { ...source.case, root: { input: source.case.input ?? null, output: source.case.observedOutput ?? null } },
      replay: { ...config.replay, ...(batch.run ? { run: { runId: batch.run.runId, caseId: slot.caseId, trial: slot.trial } } : {}) },
      config: { ...owner.agent, timeout_seconds: Math.min(owner.agent.timeout_seconds, Math.max(1, Math.floor(remaining / 1000))) },
      cwd: config.cwd, label: config.label, params: config.params });
  } catch (error) {
    slot.progress = { kind: 'failed', message: message(error) };
    await saveAllReplayState(owner.path, owner.state);
    return;
  }
  slot.execution.status = result.status;
  slot.execution.exitCode = result.exitCode;
  if (!owner.json) { process.stdout.write(result.stdout); process.stderr.write(result.stderr); }
  if (result.exitCode !== 0) slot.progress = { kind: 'failed', message: `Agent exited with code ${result.exitCode}; it will not be run again automatically.` };
  else if (!owner.api) slot.progress = { kind: 'local_complete' };
  else if (result.status.traceIds.length !== 1) slot.progress = { kind: 'failed', message: 'Upload requires exactly one workflow trace; this agent attempt will not run again automatically.' };
  else slot.progress = { kind: 'upload_pending', traceId: result.status.traceIds[0]! };
  await saveAllReplayState(owner.path, owner.state);
  await uploadPending(owner, batch, slot);
}

async function registerBatch(owner: Runner, batch: Batch) {
  if (!owner.api || batch.run) return;
  const config = owner.state.configuration;
  if (!config.grading) throw new Error('Saved grading configuration is missing.');
  const index = Math.floor(batch.offset / config.batchSize) + 1, count = owner.state.batches.length;
  const suffix = ` [${index}/${count}]`;
  batch.run = await owner.api.register({ datasetVersionId: config.versionId, label: `${config.label.slice(0, 120 - suffix.length)}${suffix}`,
    caseIds: owner.state.cases.slice(batch.offset, batch.offset + config.batchSize).map(item => item.caseId),
    trials: config.trials, params: config.params, grading: config.grading, deadlineSeconds: config.deadlineSeconds, replay: config.replay,
    selection: { kind: 'all', groupId: owner.state.groupId, datasetId: config.datasetId, digest: owner.state.selectionDigest,
      totalCases: owner.state.cases.length, batchOffset: batch.offset, batchSize: config.batchSize },
  }, batch.registrationKey);
  await saveAllReplayState(owner.path, owner.state);
}

async function execute(owner: Runner): Promise<number> {
  let error: string | undefined;
  try {
    if (owner.state.batches.some(batch => batch.slots.some(slot => slot.progress.kind === 'executing')))
      throw new Error('A previous agent command has an unknown outcome. It will not run again automatically. Inspect that attempt before starting another run.');
    for (const [index, batch] of owner.state.batches.entries()) {
      if (!owner.json) process.stderr.write(`Batch ${index + 1} of ${owner.state.batches.length}\n`);
      await registerBatch(owner, batch);
      for (const slot of batch.slots) await executeSlot(owner, batch, slot);
    }
  } catch (cause) { error = message(cause); }
  const slots = owner.state.batches.flatMap(batch => batch.slots);
  const completed = slots.filter(slot => slot.progress.kind === 'accepted' || slot.progress.kind === 'local_complete').length;
  const failed = slots.filter(slot => slot.progress.kind === 'failed').length;
  const exitCode = error || completed !== slots.length ? 1 : 0;
  const report = { exitCode, groupId: owner.state.groupId, stateFile: owner.path, selectionDigest: owner.state.selectionDigest,
    totalCases: owner.state.cases.length, completedAttempts: completed, failedAttempts: failed, pendingAttempts: slots.length - completed - failed,
    runs: owner.state.batches.flatMap((batch, index) => batch.run ? [{ batch: index + 1, runId: batch.run.runId }] : []),
    reports: owner.state.batches.flatMap((batch, index) => batch.slots.filter(slot => slot.execution).map(slot => ({
      batch: index + 1, caseId: slot.caseId, trial: slot.trial, progress: slot.progress.kind, ...slot.execution,
    }))),
    failures: slots.filter(slot => slot.progress.kind === 'failed').map(slot => ({ caseId: slot.caseId, trial: slot.trial, ...slot.progress })), ...(error ? { error } : {}) };
  if (owner.json) process.stdout.write(`${JSON.stringify(report)}\n`);
  else {
    process.stdout.write(`${completed} of ${slots.length} attempts finished. ${report.runs.length} saved batch runs.\nState: ${owner.path}\n`);
    for (const run of report.runs) process.stdout.write(`Batch ${run.batch}: ${run.runId}\n`);
    for (const attempt of report.reports) {
      const status = attempt.status;
      if (status) process.stdout.write(`${attempt.caseId}, attempt ${attempt.trial}: HIT ${status.hits}; MISS ${status.firstMiss ? 1 : 0}; LIVE ${status.liveCalls}. ${status.firstMiss ? `First miss: ${status.firstMiss.kind} ${status.firstMiss.name}. ` : ''}Traces: ${status.traceIds.join(', ') || 'none'}.\n`);
    }
    if (error) process.stderr.write(`${error}\nContinue with: neatlogs simulate local --resume ${JSON.stringify(owner.path)}\n`);
  }
  return exitCode;
}

export async function startWholeDataset(input: { path: string; configuration: AllReplayConfiguration; apiOptions: DatasetApiOptions; agent: ReturnType<typeof parseAgentConfig>; json: boolean }) {
  return withAllReplayLock(input.path, async path => {
    try { await stat(path); throw new Error('The state file already exists. Continue it with --resume, or choose a new file.'); }
    catch (error) { if (!(error instanceof Error && 'code' in error && error.code === 'ENOENT')) throw error; }
    const configuration = allReplayConfigurationSchema.parse(input.configuration), dataset = new DatasetApi(input.apiOptions);
    if (agentConfigDigest(input.agent) !== configuration.agentDigest) throw new Error('Agent command settings do not match the saved configuration.');
    const cases = await enumerateSavedDataset(dataset, configuration.datasetId, configuration.versionId);
    const state = createAllReplayState(configuration, cases);
    await saveAllReplayState(path, state, true);
    process.stderr.write(`Entire saved dataset: ${cases.length} examples in ${state.batches.length} batches. State: ${path}\n`);
    return execute({ state, path, dataset, api: configuration.grading ? new LocalRunApi(input.apiOptions) : null, agent: input.agent, json: input.json });
  });
}

export async function continueWholeDataset(input: { path: string; apiOptions: DatasetApiOptions; json: boolean }) {
  return withAllReplayLock(input.path, async path => {
    const state = await readAllReplayState(path), saved = state.configuration;
    if (resolve(process.cwd()) !== saved.cwd || uuid(input.apiOptions.projectId, 'projectId') !== saved.projectId || new URL(input.apiOptions.baseUrl).origin !== saved.baseUrl)
      throw new Error('Continue this saved run from its original working directory, project and API origin.');
    if ((await stat(saved.agentFile)).size > 64 * 1024) throw new Error('Agent config exceeds 64 KiB.');
    const agent = parseAgentConfig(await readFile(saved.agentFile, 'utf8'));
    if (agentConfigDigest(agent) !== saved.agentDigest) throw new Error('Agent command settings changed. Start a new state file.');
    return execute({ state, path, dataset: new DatasetApi(input.apiOptions), api: saved.grading ? new LocalRunApi(input.apiOptions) : null, agent, json: input.json });
  });
}
