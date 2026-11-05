import { createHash, randomUUID } from 'node:crypto';
import { existsSync, readFileSync, writeFileSync, renameSync } from 'node:fs';
import { z } from 'zod';
import { DatasetApi, json, type DatasetApiOptions, type DatasetJson } from './dataset/api.js';
import { withAllReplayLock } from './replay/all-state.js';
import { taskCredentialRefresh } from './task-credentials.js';
import { LocalRunApi } from './replay/run-api.js';
import { trace } from './core/context.js';
import { getNeatlogsProvider } from './core/provider.js';

export const LOCAL_ROUND_TRACKING_VERSION = 1;

export class Dataset {
  constructor(readonly name: string, readonly options: { versionId?: string } = {}) {}
}
export interface EvalContext {
  expected: unknown;
  metadata: Record<string, unknown>;
  trial: number;
  params: Record<string, string | number | boolean | null>;
  metrics: Record<string, number>;
  param(name: string, fallback?: string | number | boolean | null): string | number | boolean | null | undefined;
}
const gradingSchema = z.object({ formVersionId: z.string().uuid(), evaluatorIds: z.array(z.string().uuid()).min(1).max(20) }).strict();
export interface RunEvalOptions {
  dataset: Dataset | string;
  task: (input: DatasetJson, context: EvalContext) => unknown | Promise<unknown>;
  label: string;
  local?: { experimentId: string; roundId: string; variant: "A" | "B"; attempt: number };
  params?: EvalContext['params'];
  trials?: number;
  baseline?: string;
  gates?: string[];
  grading: z.infer<typeof gradingSchema>;
  checks?: string[];
  git?: { branch: string; sha?: string };
  api?: DatasetApiOptions;
  waitTimeoutMillis?: number;
  maxErrorRate?: number;
  statePath?: string;
  resume?: boolean;
  traceEvidence?: boolean;
  caseIds?: string[];
}
export interface EvalResult { runId: string; comparison: Record<string, unknown>; gate: Awaited<ReturnType<LocalRunApi['gate']>> | null; errors: number; exitCode: 0 | 1 | 4 }
const slotSchema = z.object({ key: z.string().uuid(), executing: z.boolean(), accepted: z.boolean().optional(), body: z.object({ caseId: z.string().uuid(), trial: z.number().int(), traceId: z.string().regex(/^[a-f0-9]{32}$/).optional(), output: z.json().optional(), metrics: z.record(z.string(), z.number()).optional(), error: z.object({ code: z.string(), message: z.string() }).optional() }).strict().optional() }).strict();
const dispatchSlotSchema = z.object({ batchId: z.string().uuid(), key: z.string().uuid(), accepted: z.boolean() }).strict();
const stateSchema = z.object({ projectId: z.string().uuid(), request: z.record(z.string(), z.json()), taskDigest: z.string().regex(/^[a-f0-9]{64}$/), registrationKey: z.string().uuid(), runId: z.string().uuid().nullable(), traceEvidence: z.boolean().optional(), slots: z.record(z.string(), slotSchema),
  completionKey: z.string().uuid().optional(), gateKey: z.string().uuid().optional(), gradingDispatch: z.object({ evaluationId: z.string().uuid(), recorded: dispatchSlotSchema, candidate: dispatchSlotSchema }).strict().optional() }).strict();
type State = z.infer<typeof stateSchema>;
function save(path: string | undefined, state: State) {
  if (!path) return;
  writeFileSync(path + '.tmp', JSON.stringify(state), { mode: 0o600 });
  renameSync(path + '.tmp', path);
}
const caseSchema = z.object({ caseId: z.string().uuid(), revisionId: z.string().uuid(), input: z.json(), expected: z.unknown(), metadata: z.record(z.string(), z.unknown()) }).passthrough();
async function resolve(api: DatasetApi, dataset: Dataset, requested?: string[]) {
  let id = dataset.name;
  if (!z.string().uuid().safeParse(id).success) {
    const matches = new Set<string>(), seen = new Set<string>();
    let cursor: string | null | undefined;
    do {
      const page = z.object({ datasets: z.array(z.object({ id: z.string().uuid(), name: z.string() }).passthrough()).max(500), nextCursor: z.string().uuid().nullable().optional() }).parse(await api.request('GET', `/datasets${cursor ? `?before=${cursor}` : ''}`));
      for (const item of page.datasets) if (item.name === dataset.name) matches.add(item.id);
      if (matches.size > 1) throw new Error('Use a dataset UUID or one unambiguous dataset name.');
      if ((page.nextCursor === undefined && page.datasets.length === 500) || (page.nextCursor && seen.has(page.nextCursor))) throw new Error('Dataset catalog pagination is incomplete; use a dataset UUID.');
      cursor = page.nextCursor;
      if (cursor) seen.add(cursor);
    } while (cursor);
    const [match] = matches;
    if (!match) throw new Error('Use a dataset UUID or one unambiguous dataset name.');
    id = match;
  }
  const suffix = dataset.options.versionId ? `?versionId=${z.string().uuid().parse(dataset.options.versionId)}` : '';
  const overview = z.object({ dataset: z.object({ caseCount: z.number().int().min(1).max(100000) }), version: z.object({ id: z.string().uuid() }) }).parse(await api.request('GET', `/datasets/${id}${suffix}`));
  const versionId = overview.version.id;
  if (!requested && overview.dataset.caseCount > 100) throw new Error('Select up to 100 cases for one run.');
  const entries: { caseId: string }[] = [];
  for (let offset = 0; offset < overview.dataset.caseCount; offset += 100) {
    const page = z.object({ cases: z.array(z.object({ caseId: z.string().uuid() })), total: z.number().int() }).parse(await api.request('GET', `/datasets/${id}/cases?versionId=${versionId}&offset=${offset}&limit=100`));
    if (page.total !== overview.dataset.caseCount || page.cases.length !== Math.min(100, page.total - offset)) throw new Error('Dataset enumeration is incomplete.');
    entries.push(...page.cases);
  }
  if (requested && requested.some(id => !entries.some(item => item.caseId === id))) throw new Error('Selected cases must belong to the pinned dataset.');
  const cases = [];
  for (const item of entries.filter(item => !requested || requested.includes(item.caseId))) cases.push(z.object({ case: caseSchema }).parse(await api.request('GET', `/datasets/${id}/cases/${item.caseId}?versionId=${versionId}`)).case);
  return { versionId, cases };
}

/** Resume uploads from saved outputs; an uncertain task execution is never repeated. */
export async function runEval(options: RunEvalOptions): Promise<EvalResult> {
  if (!gradingSchema.safeParse(options.grading).success) throw new Error('Select valid grading with a formVersionId and 1–20 evaluatorIds.');
  if (options.statePath) {
    return withAllReplayLock(options.statePath, statePath => runEvalLocked({ ...options, statePath }));
  }
  return runEvalLocked(options);
}

async function runEvalLocked(options: RunEvalOptions): Promise<EvalResult> {
  if (options.traceEvidence && !getNeatlogsProvider()) throw new Error('Initialize the Neatlogs SDK before running with trace evidence.');
  const trials = z.number().int().min(1).max(10).parse(options.trials ?? 1);
  const timeout = z.number().finite().positive().parse(options.waitTimeoutMillis ?? 300_000);
  const maxErrorRate = z.number().min(0).max(1).parse(options.maxErrorRate ?? 0);
  if (options.resume && !options.statePath) throw new Error('Resume requires the original statePath.');
  const selectedApi = options.api ?? { token: process.env.NEATLOGS_TOKEN ?? '', projectId: process.env.NEATLOGS_PROJECT_ID ?? '', baseUrl: process.env.NEATLOGS_HOST ?? 'https://app.neatlogs.com' };
  const apiOptions = { ...selectedApi, refreshAccessToken: selectedApi.refreshAccessToken ?? taskCredentialRefresh(selectedApi) };
  const datasetApi = new DatasetApi(apiOptions), runApi = new LocalRunApi(apiOptions);
  const saved = options.resume && options.statePath ? stateSchema.parse(JSON.parse(readFileSync(options.statePath, 'utf8'))) : null;
  const selected = typeof options.dataset === 'string' ? new Dataset(options.dataset) : options.dataset;
  const dataset = saved ? new Dataset(selected.name, { versionId: z.string().uuid().parse(saved.request.datasetVersionId) }) : selected;
  const requested = options.caseIds === undefined ? undefined : z.array(z.string().uuid()).min(1).max(100).parse(options.caseIds);
  if (requested && new Set(requested).size !== requested.length) throw new Error('Selected cases must be unique.');
  const resolved = await resolve(datasetApi, dataset, requested);
  const versionId = resolved.versionId;
  if (requested && (new Set(requested).size !== requested.length || requested.some(id => !resolved.cases.some(item => item.caseId === id)))) throw new Error('Selected cases must be unique and belong to the pinned dataset.');
  const cases = requested ? resolved.cases.filter(item => requested.includes(item.caseId)) : resolved.cases;
  const local = options.local ? z.object({ experimentId: z.string().uuid(), roundId: z.string().uuid(), variant: z.enum(["A", "B"]), attempt: z.number().int().min(1).max(100) }).strict().parse(options.local) : undefined;
  const request = { ...(local ? { local } : {}), datasetVersionId: versionId, caseIds: cases.map(item => item.caseId), label: z.string().trim().min(1).parse(options.label), params: options.params ?? {}, trials, deadlineSeconds: 900,
    grading: options.grading, ...(options.baseline ? { baseline: options.baseline } : {}), ...(options.gates ? { gates: options.gates } : {}), ...(options.git ? { git: options.git } : {}), ...(options.checks ? { checks: options.checks } : {}) };
  const taskDigest = createHash('sha256').update(options.task.toString()).digest('hex');
  const requestJson = z.record(z.string(), z.json()).parse(json(request));
  if (saved && (JSON.stringify(saved.request) !== JSON.stringify(requestJson) || saved.projectId !== apiOptions.projectId || saved.taskDigest !== taskDigest || Boolean(saved.traceEvidence) !== Boolean(options.traceEvidence))) throw new Error('Resume must use the same project, pinned dataset and run configuration.');
  if (!saved && options.statePath && existsSync(options.statePath)) throw new Error('State file exists; use resume or choose a new path.');
  const state: State = saved ?? { projectId: apiOptions.projectId, request: requestJson, taskDigest, registrationKey: randomUUID(), runId: null, traceEvidence: Boolean(options.traceEvidence), slots: {} };
  save(options.statePath, state);
  const runId = state.runId ?? (await runApi.register(request, state.registrationKey)).runId;
  const receipt = process.env.NEATLOGS_EVAL_RESULT_FILE;
  if (receipt) {
    try { writeFileSync(receipt, runId, { flag: 'wx', mode: 0o600 }); }
    catch { /* A CLI activity receipt must never change the run outcome. */ }
  }
  if (!state.runId) { state.runId = runId; save(options.statePath, state); }
  for (const item of cases) for (let trial = 1; trial <= trials; trial++) {
    const key = `${item.caseId}:${trial}`;
    let slot = state.slots[key];
    if (slot?.accepted) continue;
    if (slot?.executing) throw new Error('Task execution may have started without a saved output. Inspect it before starting a new run.');
    if (!slot) {
      slot = { key: randomUUID(), executing: true }; state.slots[key] = slot; save(options.statePath, state);
      const params = options.params ?? {}, started = performance.now();
      const context: EvalContext = { expected: item.expected, metadata: item.metadata, trial, params, metrics: {}, param: (name, fallback) => Object.hasOwn(params, name) ? params[name] : fallback };
      try {
        let traceId = '';
        const output = json(await trace({ name: options.label, kind: 'WORKFLOW', input: item.input, attributes: { 'neatlogs.origin': 'simulation', 'neatlogs.replay.run_id': runId, 'neatlogs.replay.case_id': item.caseId, 'neatlogs.replay.trial': trial } }, (span) => {
          traceId = span.spanContext().traceId;
          return options.task(item.input, context);
        }));
        if (options.traceEvidence) {
          if (!/^[a-f0-9]{32}$/.test(traceId) || /^0+$/.test(traceId)) throw new Error('Initialize the Neatlogs SDK before running with trace evidence.');
          slot.body = { caseId: item.caseId, trial, traceId };
        } else slot.body = { caseId: item.caseId, trial, output, metrics: { ...context.metrics, latencyMs: performance.now() - started } };
      } catch (error) { slot.body = { caseId: item.caseId, trial, error: { code: error instanceof Error ? error.name.slice(0, 100) : 'TASK_ERROR', message: (error instanceof Error ? error.message : String(error)).slice(0, 1000) } }; }
      slot.executing = false; save(options.statePath, state);
    }
    if (!slot.body) throw new Error('Saved output is missing.');
    if (slot.body.traceId) {
      const sdkPackage = 'neat' + 'logs';
      const sdk: unknown = await import(sdkPackage);
      const flush = z.object({ flush: z.function() }).parse(sdk).flush;
      if (!await flush()) throw new Error('Could not export the task trace. Resume after checking the SDK connection.');
      await runApi.upload(runId, { caseId: slot.body.caseId, trial: slot.body.trial, traceId: slot.body.traceId }, { key: slot.key, waitMillis: 300_000 });
    } else await runApi.taskOutput(runId, slot.body, slot.key);
    slot.accepted = true; save(options.statePath, state);
  }
  state.completionKey ??= randomUUID();
  save(options.statePath, state);
  let status = await runApi.complete(runId, state.completionKey);
  const errors = Object.values(state.slots).filter(slot => slot.body?.error).length;
  const deadline = Date.now() + timeout;
  while (options.grading && status.state === 'completed' &&
    (status.gradingHandoffState === 'pending' || status.gradingHandoffState === 'ready') && Date.now() < deadline) {
    await new Promise(resolve => setTimeout(resolve, Math.min(1000, deadline - Date.now())));
    status = await runApi.status(runId);
  }
  let comparison = await runApi.comparison(runId);
  while (comparison.state === 'pending' && Date.now() < deadline) { await new Promise(resolve => setTimeout(resolve, Math.min(1000, deadline - Date.now()))); comparison = await runApi.comparison(runId); }
  const coverage = z.object({ complete: z.boolean() }).parse(comparison.coverage);
  if (errors / (cases.length * trials) > maxErrorRate || !coverage.complete) return { runId, comparison, gate: null, errors, exitCode: 4 };
  if (!options.gates?.length) return { runId, comparison, gate: null, errors, exitCode: 0 };
  state.gateKey ??= randomUUID();
  save(options.statePath, state);
  const gate = await runApi.gate(runId, state.gateKey);
  return { runId, comparison, gate, errors, exitCode: gate.verdict === 'pass' ? 0 : gate.verdict === 'fail' ? 1 : 4 };
}
