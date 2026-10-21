import { createHash, randomUUID } from 'node:crypto';
import { existsSync, readFileSync, writeFileSync, renameSync } from 'node:fs';
import { z } from 'zod';
import { DatasetApi, json, type DatasetApiOptions, type DatasetJson } from './dataset/api.js';
import { LocalRunApi } from './replay/run-api.js';
import { trace } from './core/context.js';

export class Dataset {
  constructor(readonly name: string, readonly options: { versionId?: string; tag?: string } = {}) {}
}
export interface EvalContext {
  expected: unknown;
  metadata: Record<string, unknown>;
  trial: number;
  params: Record<string, string | number | boolean | null>;
  metrics: Record<string, number>;
  param(name: string, fallback?: string | number | boolean | null): string | number | boolean | null | undefined;
}
export interface RunEvalOptions {
  dataset: Dataset | string;
  task: (input: DatasetJson, context: EvalContext) => unknown | Promise<unknown>;
  label: string;
  params?: EvalContext['params'];
  trials?: number;
  baseline?: string;
  gates?: string[];
  grading?: { formVersionId: string; evaluatorIds: string[] };
  checks?: string[];
  git?: { branch: string; sha?: string };
  api?: DatasetApiOptions;
  waitTimeoutMillis?: number;
  maxErrorRate?: number;
  statePath?: string;
  resume?: boolean;
}
export interface EvalResult { runId: string; comparison: Record<string, unknown>; gate: Awaited<ReturnType<LocalRunApi['gate']>> | null; errors: number; exitCode: 0 | 1 | 4 }
const slotSchema = z.object({ key: z.string().uuid(), executing: z.boolean(), accepted: z.boolean().optional(), body: z.object({ caseId: z.string().uuid(), trial: z.number().int(), output: z.json().optional(), metrics: z.record(z.string(), z.number()).optional(), error: z.object({ code: z.string(), message: z.string() }).optional() }).strict().optional() }).strict();
const stateSchema = z.object({ projectId: z.string().uuid(), request: z.record(z.string(), z.json()), taskDigest: z.string().regex(/^[a-f0-9]{64}$/), registrationKey: z.string().uuid(), runId: z.string().uuid().nullable(), slots: z.record(z.string(), slotSchema) }).strict();
type State = z.infer<typeof stateSchema>;
function save(path: string | undefined, state: State) {
  if (!path) return;
  writeFileSync(path + '.tmp', JSON.stringify(state), { mode: 0o600 });
  renameSync(path + '.tmp', path);
}
const caseSchema = z.object({ caseId: z.string().uuid(), revisionId: z.string().uuid(), input: z.json(), expected: z.unknown(), metadata: z.record(z.string(), z.unknown()) }).passthrough();
async function resolve(api: DatasetApi, dataset: Dataset) {
  if (dataset.options.tag && dataset.options.versionId) throw new Error('Choose a dataset tag or versionId.');
  let id = dataset.name;
  if (!z.string().uuid().safeParse(id).success) {
    const list = z.object({ datasets: z.array(z.object({ id: z.string().uuid(), name: z.string() }).passthrough()) }).parse(await api.request('GET', '/datasets'));
    const matches = list.datasets.filter(item => item.name === dataset.name);
    if (matches.length !== 1 || !matches[0]) throw new Error('Use a dataset UUID or one unambiguous dataset name.');
    id = matches[0].id;
  }
  const suffix = dataset.options.versionId ? `?versionId=${z.string().uuid().parse(dataset.options.versionId)}` : dataset.options.tag ? `?tag=${encodeURIComponent(dataset.options.tag)}` : '';
  const overview = z.object({ dataset: z.object({ caseCount: z.number().int().min(1).max(100) }), version: z.object({ id: z.string().uuid() }) }).parse(await api.request('GET', `/datasets/${id}${suffix}`));
  const versionId = overview.version.id;
  const page = z.object({ cases: z.array(z.object({ caseId: z.string().uuid() })), total: z.number().int() }).parse(await api.request('GET', `/datasets/${id}/cases?versionId=${versionId}&offset=0&limit=100`));
  if (page.total !== overview.dataset.caseCount || page.cases.length !== page.total) throw new Error('Dataset enumeration is incomplete.');
  const cases = [];
  for (const item of page.cases) cases.push(z.object({ case: caseSchema }).parse(await api.request('GET', `/datasets/${id}/cases/${item.caseId}?versionId=${versionId}`)).case);
  return { versionId, cases };
}

/** Resume uploads from saved outputs; an uncertain task execution is never repeated. */
export async function runEval(options: RunEvalOptions): Promise<EvalResult> {
  const trials = z.number().int().min(1).max(10).parse(options.trials ?? 1);
  const timeout = z.number().finite().positive().parse(options.waitTimeoutMillis ?? 300_000);
  const maxErrorRate = z.number().min(0).max(1).parse(options.maxErrorRate ?? 0);
  if (options.resume && !options.statePath) throw new Error('Resume requires the original statePath.');
  const apiOptions = options.api ?? { token: process.env.NEATLOGS_TOKEN ?? '', projectId: process.env.NEATLOGS_PROJECT_ID ?? '', baseUrl: process.env.NEATLOGS_HOST ?? 'https://app.neatlogs.com' };
  const datasetApi = new DatasetApi(apiOptions), runApi = new LocalRunApi(apiOptions);
  const saved = options.resume && options.statePath ? stateSchema.parse(JSON.parse(readFileSync(options.statePath, 'utf8'))) : null;
  const selected = typeof options.dataset === 'string' ? new Dataset(options.dataset) : options.dataset;
  const dataset = saved ? new Dataset(selected.name, { versionId: z.string().uuid().parse(saved.request.datasetVersionId) }) : selected;
  const { versionId, cases } = await resolve(datasetApi, dataset);
  const request = { datasetVersionId: versionId, caseIds: cases.map(item => item.caseId), label: z.string().trim().min(1).parse(options.label), params: options.params ?? {}, trials, deadlineSeconds: 900,
    ...(options.grading ? { grading: options.grading } : {}), ...(options.baseline ? { baseline: options.baseline } : {}), ...(options.gates ? { gates: options.gates } : {}), ...(options.git ? { git: options.git } : {}), ...(options.checks ? { checks: options.checks } : {}) };
  const taskDigest = createHash('sha256').update(options.task.toString()).digest('hex');
  const requestJson = z.record(z.string(), z.json()).parse(json(request));
  if (saved && (JSON.stringify(saved.request) !== JSON.stringify(requestJson) || saved.projectId !== apiOptions.projectId || saved.taskDigest !== taskDigest)) throw new Error('Resume must use the same project, pinned dataset and run configuration.');
  if (!saved && options.statePath && existsSync(options.statePath)) throw new Error('State file exists; use resume or choose a new path.');
  const state: State = saved ?? { projectId: apiOptions.projectId, request: requestJson, taskDigest, registrationKey: randomUUID(), runId: null, slots: {} };
  save(options.statePath, state);
  if (!state.runId) { state.runId = (await runApi.register(request, state.registrationKey)).runId; save(options.statePath, state); }
  const runId = state.runId;
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
        const output = json(await trace({ name: options.label, kind: 'WORKFLOW', input: item.input, attributes: { 'neatlogs.origin': 'simulation', 'neatlogs.replay.run_id': runId, 'neatlogs.replay.case_id': item.caseId, 'neatlogs.replay.trial': trial } }, () => options.task(item.input, context)));
        slot.body = { caseId: item.caseId, trial, output, metrics: { ...context.metrics, latencyMs: performance.now() - started } };
      } catch (error) { slot.body = { caseId: item.caseId, trial, error: { code: error instanceof Error ? error.name.slice(0, 100) : 'TASK_ERROR', message: (error instanceof Error ? error.message : String(error)).slice(0, 1000) } }; }
      slot.executing = false; save(options.statePath, state);
    }
    if (!slot.body) throw new Error('Saved output is missing.');
    await runApi.taskOutput(runId, slot.body, slot.key); slot.accepted = true; save(options.statePath, state);
  }
  await runApi.complete(runId);
  const errors = Object.values(state.slots).filter(slot => slot.body?.error).length;
  const deadline = Date.now() + timeout;
  let comparison = await runApi.comparison(runId);
  while (comparison.state === 'pending' && Date.now() < deadline) { await new Promise(resolve => setTimeout(resolve, Math.min(1000, deadline - Date.now()))); comparison = await runApi.comparison(runId); }
  const coverage = z.object({ complete: z.boolean() }).parse(comparison.coverage);
  if (errors / (cases.length * trials) > maxErrorRate || !coverage.complete) return { runId, comparison, gate: null, errors, exitCode: 4 };
  const gate = await runApi.gate(runId);
  return { runId, comparison, gate, errors, exitCode: gate.verdict === 'pass' ? 0 : gate.verdict === 'fail' ? 1 : 4 };
}
