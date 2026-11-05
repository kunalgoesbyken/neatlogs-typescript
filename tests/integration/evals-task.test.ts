import { createServer, type Server } from 'node:http';
import { randomUUID } from 'node:crypto';
import { mkdtempSync, readFileSync, rmSync, statSync, symlinkSync, lstatSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { Dataset, runEval } from '../../src/evals.js';
import { Client } from '../../src/core/client.js';
import { trace } from '../../src/core/context.js';
import { span } from '../../src/decorators/index.js';
import { LocalRunApi } from '../../src/replay/run-api.js';
import type { ReadableSpan } from '@opentelemetry/sdk-trace-base';

const [projectId, datasetId, versionId, caseId, revisionId, runId] = Array.from({ length: 6 }, () => randomUUID());
let server: Server;
let baseUrl: string;
let directory: string;
let failUpload = false;
let verdict = 'pass';
let uploads: Record<string, unknown>[];
let registrations: Record<string, unknown>[];
let paths: string[];
let catalogHeaders: (string | undefined)[];
let manyCases: string[];
let datasetPages: { datasets: { id: string; name: string }[]; nextCursor?: string | null }[];
const grading = { formVersionId: randomUUID(), evaluatorIds: [randomUUID()] };
let comparisonsUntilReady = 0;
let handoffReady = false;
let completionKeys: string[];
let failCompletionCount = 0;
let gateKeys: string[];
let failGateCount = 0;
let comparisonRows: { key: string }[] = [];
let exported: ReadableSpan[] = [];
let evidenceAtUpload: ReadableSpan[][] = [];
let caseInput = 'hello', rejectEvidence = false, evidenceRejectionCode: string | undefined, failRejectedFailure = false, disconnectUpload = false;
let uploadKeys: string[] = [];
const clients: Client[] = [];
beforeEach(async () => {
  caseInput = 'hello'; rejectEvidence = false; evidenceRejectionCode = 'GRADING_EVIDENCE_LIMIT'; failRejectedFailure = false; disconnectUpload = false; uploadKeys = [];
  comparisonRows = []; exported = []; evidenceAtUpload = [];
  registrations = []; uploads = []; paths = []; catalogHeaders = []; manyCases = []; datasetPages = []; completionKeys = []; gateKeys = []; comparisonsUntilReady = 0; handoffReady = false; failUpload = false; failCompletionCount = 0; failGateCount = 0; verdict = 'pass'; directory = mkdtempSync(join(tmpdir(), 'neatlogs-eval-'));
  server = createServer(async (request, response) => {
    const path = request.url ?? ''; paths.push(path);
    const chunks = []; for await (const chunk of request) chunks.push(Buffer.from(chunk));
    const body = chunks.length ? JSON.parse(Buffer.concat(chunks).toString()) : {};
    let data: unknown, status = 200, code: string | undefined;
    if (request.method === 'GET') {
      if (new URL(path, 'http://localhost').pathname === '/api/v1/public/datasets') {
        catalogHeaders.push(request.headers.prefer);
        expect(request.headers.authorization).toBe('Bearer task-token');
        expect(request.headers['x-project-id']).toBe(projectId);
        const before = new URL(path, 'http://localhost').searchParams.get('before');
        data = datasetPages[before ? 1 : 0];
      }
      else if (path.includes('/comparison')) {
        const complete = comparisonsUntilReady-- <= 0;
        const after = new URL(path, 'http://localhost').searchParams.get('after');
        const start = after ? comparisonRows.findIndex(row => row.key === after) + 1 : 0;
        const limit = Number(new URL(path, 'http://localhost').searchParams.get('limit') ?? 30);
        const page = comparisonRows.slice(start, start + limit);
        data = { state: complete ? 'ready' : 'pending', coverage: { complete }, ...(comparisonRows.length ? { pairs: page, nextCursor: start + page.length < comparisonRows.length ? page.at(-1)?.key : null, questionSummaries: [{ questionId: 'quality', total: comparisonRows.length }] } : {}) };
      }
      else if (path.endsWith(`/runs/${runId}`)) data = { runId, state: 'completed', deadlineAt: '2030-01-01T00:00:00Z', itemCount: 1, completedCount: 1, failedCount: 0, pendingCount: 0, gradingHandoffState: 'dispatched' };
      else if (path.includes('/cases/') && (path.includes(`/cases/${caseId}`) || manyCases.some(id => path.includes(`/cases/${id}`)))) data = { case: { caseId: path.split('/cases/')[1]!.split('?')[0], revisionId, input: caseInput, expected: { value: 'hello' }, metadata: { language: 'en' } } };
      else if (path.includes('/cases?')) { const offset = Number(new URL(path, 'http://localhost').searchParams.get('offset')); data = manyCases.length ? { cases: manyCases.slice(offset, offset + 100).map(caseId => ({ caseId })), total: manyCases.length } : { cases: [{ caseId }], total: 1 }; }
      else data = { dataset: { caseCount: manyCases.length || 1 }, version: { id: versionId } };
    } else {
      expect(request.headers.authorization).toBe('Bearer task-token'); expect(request.headers['x-project-id']).toBe(projectId); expect(request.headers['idempotency-key']).toBeTruthy();
      if (path.endsWith('/complete')) { completionKeys.push(String(request.headers['idempotency-key'])); if (failCompletionCount > 0) { status = 503; failCompletionCount--; } data = { runId, state: 'completed', deadlineAt: '2030-01-01T00:00:00Z', itemCount: 2, completedCount: 2, failedCount: 0, pendingCount: 0, gradingHandoffState: handoffReady ? 'ready' : 'dispatched' }; }
      else if (path.endsWith('/outputs')) {
        uploadKeys.push(String(request.headers['idempotency-key']));
        if (disconnectUpload) { request.socket.destroy(); return; }
        if (rejectEvidence && 'output' in body) { status = 413; code = evidenceRejectionCode; }
        else if (failUpload || failRejectedFailure && body.error) status = 503;
        else uploads.push(body);
        data = body.traceId ? { runId, itemId: randomUUID(), outputId: randomUUID(), traceId: body.traceId, state: 'succeeded' } : { outputId: randomUUID(), state: body.error ? 'failed' : 'succeeded' };
        if (body.traceId) evidenceAtUpload.push([...exported]);
      }
      else if (path.endsWith('/gate')) { gateKeys.push(String(request.headers['idempotency-key'])); if (failGateCount > 0) { status = 503; failGateCount--; } data = { verdict }; }
      else data = { runId, state: path.endsWith('/complete') ? 'completed' : 'queued', deadlineAt: '2030-01-01T00:00:00Z', itemCount: 2, completedCount: 2, failedCount: 0, pendingCount: 0, gradingHandoffState: 'dispatched' };
      if (!path.endsWith('/complete') && !path.endsWith('/gate') && !path.endsWith('/outputs')) {
        registrations.push(body);
        const { completedCount: _completed, failedCount: _failed, pendingCount: _pending, gradingHandoffState: _handoff, ...receipt } = data as Record<string, unknown>; data = receipt;
      }
    }
    response.writeHead(status, { 'Content-Type': 'application/json' }); response.end(JSON.stringify({ success: status < 400, ...(code ? { code } : {}), data }));
  });
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  const address = server.address(); if (!address || typeof address === 'string') throw new Error('No local test address'); baseUrl = `http://127.0.0.1:${address.port}`;
});
afterEach(async () => { await Promise.all(clients.splice(0).map(client => client.shutdown())); vi.unstubAllEnvs(); await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve())); rmSync(directory, { recursive: true, force: true }); });

describe('task evaluation over the public HTTP contract', () => {
  it.each([undefined, { formVersionId: randomUUID(), evaluatorIds: [] }])('rejects missing or empty grading before any request', async grading => {
    const task = vi.fn(() => 'answer');
    await expect(runEval({ dataset: datasetId!, label: 'missing grading', grading, api: { projectId: projectId!, token: 'task-token', baseUrl }, task })).rejects.toThrow(/grading/i);
    expect(paths).toEqual([]);
    expect(task).not.toHaveBeenCalled();
  });
  it('rejects nonzero error tolerance before task, state or API work for graded runs', async () => {
    const task = vi.fn(() => 'answer'), statePath = join(directory, 'unsupported-tolerance.json');
    await expect(runEval({ grading, dataset: datasetId!, label: 'tolerance', maxErrorRate: 0.1, statePath, api: { projectId: projectId!, token: 'task-token', baseUrl }, task })).rejects.toThrow(/maxErrorRate must be zero/);
    expect(paths).toEqual([]);
    expect(task).not.toHaveBeenCalled();
    expect(() => statSync(statePath)).toThrow();
  });
  it.each([false, true])('finds older dataset names and rejects later-page duplicates: %s', async duplicate => {
    const first = Array.from({ length: 500 }, (_, index) => ({ id: randomUUID(), name: duplicate && index === 0 ? 'support' : `other-${index}` }));
    const cursor = first.at(-1)!.id;
    datasetPages = [{ datasets: first, nextCursor: cursor }, { datasets: [{ id: datasetId!, name: 'support' }], nextCursor: null }];
    const result = runEval({ grading, dataset: 'support', label: 'named', api: { projectId: projectId!, token: 'task-token', baseUrl }, task: () => 'answer' });
    if (duplicate) await expect(result).rejects.toThrow(/unambiguous/);
    else expect((await result).exitCode).toBe(0);
    expect(paths.slice(0, 2)).toEqual(['/api/v1/public/datasets', `/api/v1/public/datasets?before=${cursor}`]);
    expect(catalogHeaders).toEqual(['pagination=cursor', 'pagination=cursor']);
    expect(uploads).toHaveLength(duplicate ? 0 : 1);
  });
  it('accepts a complete legacy catalog without a cursor field', async () => {
    datasetPages = [{ datasets: [{ id: datasetId!, name: 'support' }] }];
    const result = await runEval({ grading, dataset: 'support', label: 'legacy catalog', api: { projectId: projectId!, token: 'task-token', baseUrl }, task: () => 'answer' });
    expect(result.exitCode).toBe(0);
    expect(uploads).toHaveLength(1);
    expect(catalogHeaders).toEqual(['pagination=cursor']);
  });
  it.each(['missing', 'repeated'])('rejects %s catalog continuation rather than silently truncating', async failure => {
    const first = Array.from({ length: 500 }, (_, index) => ({ id: randomUUID(), name: index === 0 ? 'support' : 'other' }));
    const cursor = first.at(-1)!.id;
    datasetPages = failure === 'missing' ? [{ datasets: first }] : [{ datasets: first, nextCursor: cursor }, { datasets: [], nextCursor: cursor }];
    await expect(runEval({ grading, dataset: 'support', label: 'named', api: { projectId: projectId!, token: 'task-token', baseUrl }, task: () => 'answer' })).rejects.toThrow(/pagination/i);
    expect(uploads).toEqual([]);
  });

  it('pins round and variant identity across resume and rejects switching variants', async () => {
    const local = { experimentId: randomUUID(), roundId: randomUUID(), variant: 'B' as const, attempt: 2 };
    const task = vi.fn(() => 'answer');
    const options = { grading, dataset: datasetId!, label: 'round B retry', local, statePath: join(directory, 'round.json'), api: { projectId: projectId!, token: 'task-token', baseUrl }, task };
    await runEval(options);
    expect(registrations[0]?.local).toEqual(local);
    await runEval({ ...options, resume: true });
    expect(task).toHaveBeenCalledTimes(1);
    expect(registrations).toHaveLength(1);
    await expect(runEval({ ...options, local: { ...local, variant: 'A' }, resume: true })).rejects.toThrow(/same project/);
  });
  it('completes descriptive runs without issuing an empty quality gate', async () => {
    const result = await runEval({ grading, dataset: datasetId!, label: 'descriptive', api: { projectId: projectId!, token: 'task-token', baseUrl }, task: () => 'answer' });
    expect(result.exitCode).toBe(0);
    expect(result.gate).toBeNull();
    expect(gateKeys).toEqual([]);
  });
  it.each(['saved', `run:${randomUUID()}`])('exposes remaining comparison pages with whole-run summaries for %s baselines', async baseline => {
    manyCases = Array.from({ length: 31 }, () => randomUUID());
    comparisonRows = manyCases.map(key => ({ key }));
    const result = await runEval({ grading, dataset: datasetId!, label: 'paged', baseline, api: { projectId: projectId!, token: 'task-token', baseUrl }, task: () => 'answer' });
    expect(uploads).toHaveLength(31);
    expect(result.comparison.pairs).toHaveLength(30);
    expect(result.comparison.nextCursor).toBe(manyCases[29]);
    const api = new LocalRunApi({ projectId: projectId!, token: 'task-token', baseUrl });
    const next = await api.comparison(result.runId, { baseline, after: result.comparison.nextCursor!, limit: 100 });
    expect(next.pairs).toEqual([{ key: manyCases[30] }]);
    expect(next.nextCursor).toBeNull();
    expect(next.questionSummaries).toEqual(result.comparison.questionSummaries);
    expect(next.coverage).toEqual({ complete: true });
    expect(paths.at(-1)).toContain(`baseline=${encodeURIComponent(baseline)}`);
  });

  it.each([false, true])('exports isolated root evidence and completed masking from an initialized Client, nested=%s', async nested => {
    const client = new Client({ apiKey: 'test-ingest', workflowName: 'eval-test', captureLogs: false, flushInterval: 600,
      mask: async data => { await Promise.resolve(); data.attributes['test.mask.complete'] = true; return data; },
      spanExporter: { export(spans, callback) { exported.push(...spans); callback({ code: 0 }); }, async shutdown() {}, async forceFlush() {} },
    });
    clients.push(client);
    const lookup = span({ kind: 'TOOL', name: 'lookup' }, async (value: unknown) => ({ answer: value }));
    let outerId: string | undefined;
    const run = () => runEval({ grading, dataset: datasetId!, label: 'isolated-task', traceEvidence: true, api: { projectId: projectId!, token: 'task-token', baseUrl }, task: lookup });
    const result = await client.activate(() => nested ? trace({ name: 'caller', kind: 'WORKFLOW' }, parent => { outerId = parent.spanContext().traceId; return run(); }) : run());
    expect(result.exitCode).toBe(0);
    const root = evidenceAtUpload[0]?.find(item => item.name === 'isolated-task');
    expect(root?.parentSpanId).toBeUndefined();
    expect(root?.spanContext().traceId).toBe(uploads[0]?.traceId);
    if (nested) expect(root?.spanContext().traceId).not.toBe(outerId);
    expect(root?.attributes).toMatchObject({ 'neatlogs.replay.run_id': runId, 'neatlogs.replay.case_id': caseId, 'neatlogs.replay.trial': 1, 'test.mask.complete': true });
    expect(evidenceAtUpload[0]?.some(item => item.name === 'lookup')).toBe(true);
    expect(evidenceAtUpload[0]?.some(item => item.attributes['neatlogs.trace.complete'] === true)).toBe(true);
  });

  it.each([false, true])('seals oversized outputs and resumes current/legacy delivery without repeating tasks, legacy=%s', async legacy => {
    const task = vi.fn().mockImplementationOnce(() => 'x'.repeat(70 * 1024)).mockReturnValue('answer');
    const options = { grading, trials: 2, dataset: datasetId!, label: 'oversized', statePath: join(directory, 'oversized.json'), api: { projectId: projectId!, token: 'task-token', baseUrl }, task };
    failUpload = true;
    await expect(runEval(options)).rejects.toThrow(/HTTP 503/);
    const saved = JSON.parse(readFileSync(options.statePath, 'utf8'));
    expect(saved.slots[`${caseId}:1`].body).toMatchObject({ error: { code: 'OUTPUT_TOO_LARGE' } });
    expect(saved.slots[`${caseId}:1`].body).not.toHaveProperty('output');
    if (legacy) {
      saved.slots[`${caseId}:1`].body = { caseId, trial: 1, output: 'x'.repeat(70 * 1024) };
      writeFileSync(options.statePath, JSON.stringify(saved));
    }
    failUpload = false;
    const result = await runEval({ ...options, resume: true });
    expect(task).toHaveBeenCalledTimes(2);
    expect(result.exitCode).toBe(4);
    expect(uploads[0]).toMatchObject({ error: { code: 'OUTPUT_TOO_LARGE' } });
    expect(uploads[1]).toMatchObject({ trial: 2, output: 'answer' });
    expect(completionKeys).toHaveLength(1);
  });

  it.each(['ascii', 'emoji', 'nonfinite'])('saves an uploadable failure for invalid %s metrics and resumes without re-executing', async defect => {
    const task = vi.fn((_input, context) => { context.metrics[defect === 'emoji' ? '😀'.repeat(51) : defect === 'ascii' ? 'x'.repeat(101) : 'score'] = defect === 'nonfinite' ? NaN : 1; return 'answer'; });
    const options = { grading, dataset: datasetId!, label: 'invalid metrics', statePath: join(directory, 'metrics.json'), api: { projectId: projectId!, token: 'task-token', baseUrl }, task };
    failUpload = true;
    await expect(runEval(options)).rejects.toThrow(/HTTP 503/);
    const saved = JSON.parse(readFileSync(options.statePath, 'utf8'));
    expect(saved.slots[`${caseId}:1`].body).toMatchObject({ error: { code: 'INVALID_TASK_OUTPUT' } });
    expect(saved.slots[`${caseId}:1`].body).not.toHaveProperty('output');
    failUpload = false;
    expect((await runEval({ ...options, resume: true })).exitCode).toBe(4);
    expect(task).toHaveBeenCalledOnce();
    expect(uploads).toHaveLength(1);
    expect(uploads[0]).toMatchObject({ error: { code: 'INVALID_TASK_OUTPUT' } });
    expect(completionKeys).toHaveLength(1);
  });

  it('retains computed output and resumes a definite combined-evidence rejection with a new durable failure key', async () => {
    caseInput = 'x'.repeat(480_000); rejectEvidence = true; failRejectedFailure = true;
    const output = 'y'.repeat(30_000), task = vi.fn(() => output);
    const options = { grading, dataset: datasetId!, label: 'combined evidence', statePath: join(directory, 'combined.json'), api: { projectId: projectId!, token: 'task-token', baseUrl }, task };
    await expect(runEval(options)).rejects.toThrow(/HTTP 503/);
    const slot = JSON.parse(readFileSync(options.statePath, 'utf8')).slots[`${caseId}:1`];
    expect(slot.body.output).toBe(output);
    expect(slot.rejection).toMatchObject({ error: { code: 'GRADING_EVIDENCE_LIMIT' } });
    expect(slot.rejection.key).not.toBe(slot.key);
    expect(uploadKeys).toEqual([slot.key, slot.rejection.key, slot.rejection.key]);
    failRejectedFailure = false;
    expect((await runEval({ ...options, resume: true })).exitCode).toBe(4);
    expect(task).toHaveBeenCalledOnce();
    expect(uploadKeys.at(-1)).toBe(slot.rejection.key);
    expect(uploads).toHaveLength(1);
    expect(uploads[0]).toMatchObject({ error: { code: 'GRADING_EVIDENCE_LIMIT' } });
    expect(uploads[0]).not.toHaveProperty('output');
    expect(JSON.parse(readFileSync(options.statePath, 'utf8')).slots[`${caseId}:1`].body.output).toBe(output);
  });

  it('retains the original output and key after an ambiguous connection failure', async () => {
    disconnectUpload = true;
    const task = vi.fn(() => 'saved answer');
    const options = { grading, dataset: datasetId!, label: 'unknown delivery', statePath: join(directory, 'unknown.json'), api: { projectId: projectId!, token: 'task-token', baseUrl }, task };
    await expect(runEval(options)).rejects.toThrow(/request failed/);
    const slot = JSON.parse(readFileSync(options.statePath, 'utf8')).slots[`${caseId}:1`];
    expect(slot.body.output).toBe('saved answer');
    expect(slot).not.toHaveProperty('rejection');
    expect(new Set(uploadKeys)).toEqual(new Set([slot.key]));
    disconnectUpload = false;
    expect((await runEval({ ...options, resume: true })).exitCode).toBe(0);
    expect(task).toHaveBeenCalledOnce();
    expect(new Set(uploadKeys)).toEqual(new Set([slot.key]));
  });

  it.each([undefined, 'PROXY_BODY_LIMIT'])('does not replace an output after an unrecognized413: %s', async code => {
    rejectEvidence = true; evidenceRejectionCode = code;
    const task = vi.fn(() => 'saved answer');
    const options = { grading, dataset: datasetId!, label: 'unrecognized rejection', statePath: join(directory, 'unrecognized.json'), api: { projectId: projectId!, token: 'task-token', baseUrl }, task };
    await expect(runEval(options)).rejects.toThrow(/HTTP 413/);
    const slot = JSON.parse(readFileSync(options.statePath, 'utf8')).slots[`${caseId}:1`];
    expect(slot.body.output).toBe('saved answer');
    expect(slot).not.toHaveProperty('rejection');
    expect(uploadKeys).toEqual([slot.key]);
    rejectEvidence = false;
    expect((await runEval({ ...options, resume: true })).exitCode).toBe(0);
    expect(task).toHaveBeenCalledOnce();
    expect(uploadKeys).toEqual([slot.key, slot.key]);
  });

  it('saves private state atomically without following a pre-existing fixed temporary symlink', async () => {
    const statePath = join(directory, 'private.json'), victim = join(directory, 'victim');
    writeFileSync(victim, 'unchanged', { mode: 0o644 });
    symlinkSync(victim, statePath + '.tmp');
    const result = await runEval({ grading, dataset: datasetId!, label: 'private state', statePath, api: { projectId: projectId!, token: 'task-token', baseUrl }, task: () => 'answer' });
    expect(result.exitCode).toBe(0);
    expect(readFileSync(victim, 'utf8')).toBe('unchanged');
    expect(statSync(victim).mode & 0o777).toBe(0o644);
    expect(lstatSync(statePath).isFile()).toBe(true);
    expect(statSync(statePath).mode & 0o077).toBe(0);
    expect(JSON.parse(readFileSync(statePath, 'utf8')).slots[`${caseId}:1`].accepted).toBe(true);
  });

  it('resumes uncertain run completion with its original idempotency key', async () => {
    const options = { grading, dataset: datasetId!, label: 'resume completion', statePath: join(directory, 'completion.json'), api: { projectId: projectId!, token: 'task-token', baseUrl }, task: () => 'answer' };
    failCompletionCount = 2;
    await expect(runEval(options)).rejects.toThrow(/HTTP 503/);
    await runEval({ ...options, resume: true });
    expect(completionKeys).toHaveLength(3);
    expect(new Set(completionKeys).size).toBe(1);
    expect(uploads).toHaveLength(1);
  });
  it('resumes an uncertain gate with its original idempotency key', async () => {
    const options = { grading, dataset: datasetId!, label: 'resume gate', gates: ['error_rate <= 0'], statePath: join(directory, 'gate.json'), api: { projectId: projectId!, token: 'task-token', baseUrl }, task: () => 'answer' };
    failGateCount = 2;
    await expect(runEval(options)).rejects.toThrow(/HTTP 503/);
    const result = await runEval({ ...options, resume: true });
    expect(result.exitCode).toBe(0);
    expect(gateKeys).toHaveLength(3);
    expect(new Set(gateKeys).size).toBe(1);
    expect(uploads).toHaveLength(1);
  });
  it('waits for server-managed grades without dispatching duplicate work', async () => {
    comparisonsUntilReady = 1;
    handoffReady = true;
    const result = await runEval({ dataset: datasetId!, label: 'graded', grading: { formVersionId: randomUUID(), evaluatorIds: [randomUUID()] }, statePath: join(directory, 'graded.json'), api: { projectId: projectId!, token: 'task-token', baseUrl }, task: () => 'answer' });
    expect(result.exitCode).toBe(0);
    expect(paths.filter(path => path.includes('/comparison'))).toHaveLength(2);
    expect(paths.findIndex(path => path.endsWith(`/runs/${runId}`))).toBeLessThan(paths.findIndex(path => path.includes('/comparison')));
    expect(paths.some(path => path.endsWith('/ai-dispatches'))).toBe(false);
  });
  it('runs each trial, carries context and maps failed gates to exit1', async () => {
    verdict = 'fail'; const seen: number[] = [];
    const result = await runEval({ grading, dataset: new Dataset(datasetId!, { versionId }), label: 'candidate', params: { model: 'small' }, trials: 2, gates: ['new_failures <= 0'], api: { projectId: projectId!, token: 'task-token', baseUrl }, task: (_input, context) => { seen.push(context.trial); expect(context.param('model')).toBe('small'); context.metrics.costUsd = 0.001; return { answer: 'hello' }; } });
    expect(result.exitCode).toBe(1); expect(seen).toEqual([1, 2]); expect(uploads).toHaveLength(2); expect(paths).toContain(`/api/v1/public/datasets/${datasetId}?versionId=${versionId}`);
  });
  it('resumes a saved upload without repeating a completed task', async () => {
    const seen: number[] = [], statePath = join(directory, 'run.json');
    const receipt = join(directory, 'receipt');
    vi.stubEnv('NEATLOGS_EVAL_RESULT_FILE', receipt);
    const options = { grading, dataset: datasetId!, label: 'resume', trials: 2, statePath, api: { projectId: projectId!, token: 'task-token', baseUrl }, task: (_input: unknown, context: { trial: number }) => { seen.push(context.trial); return 'saved output'; } };
    failUpload = true; await expect(runEval(options)).rejects.toThrow(/HTTP 503/); expect(seen).toEqual([1]); expect(statSync(statePath).mode & 0o077).toBe(0);
    expect(readFileSync(receipt, 'utf8')).toBe(runId);
    expect(statSync(receipt).mode & 0o077).toBe(0);
    failUpload = false; const result = await runEval({ ...options, resume: true }); expect(result.exitCode).toBe(0); expect(seen).toEqual([1, 2]); expect(uploads).toHaveLength(2);
  });
  it('retains task errors and returns incomplete', async () => {
    const result = await runEval({ grading, dataset: datasetId!, label: 'error', api: { projectId: projectId!, token: 'task-token', baseUrl }, task: () => { throw new Error('task failed'); } });
    expect(result.exitCode).toBe(4); expect(result.errors).toBe(1); expect(uploads[0]?.error).toEqual({ code: 'Error', message: 'task failed' });
  });
  it('does not report trace evidence when the SDK is not initialized', async () => {
    const task = () => { throw new Error('must not run'); };
    await expect(runEval({ grading, dataset: datasetId!, label: 'trace', traceEvidence: true, api: { projectId: projectId!, token: 'task-token', baseUrl }, task })).rejects.toThrow(/Initialize the Neatlogs SDK/);
    expect(uploads).toEqual([]);
  });
  it('rejects cases outside the selected dataset before starting a run', async () => {
    await expect(runEval({ grading, dataset: datasetId!, caseIds: [randomUUID()], label: 'selection', api: { projectId: projectId!, token: 'task-token', baseUrl }, task: () => 'answer' })).rejects.toThrow(/belong to the pinned dataset/);
    expect(paths.some(path => path.endsWith('/runs'))).toBe(false);
  });
  it('runs a selected case from a later dataset page', async () => {
    manyCases = Array.from({ length: 101 }, () => randomUUID());
    const result = await runEval({ grading, dataset: datasetId!, caseIds: [manyCases[100]!], label: 'selection', api: { projectId: projectId!, token: 'task-token', baseUrl }, task: () => 'answer' });
    expect(result.exitCode).toBe(0); expect(uploads).toHaveLength(1);
    expect(paths.some(path => path.includes('offset=100'))).toBe(true);
  });
  it('rejects changed task code on resume', async () => {
    const options = { grading, dataset: datasetId!, label: 'resume', statePath: join(directory, 'run.json'), api: { projectId: projectId!, token: 'task-token', baseUrl }, task: () => 'original' };
    failUpload = true; await expect(runEval(options)).rejects.toThrow(/HTTP 503/);
    failUpload = false; await expect(runEval({ ...options, resume: true, task: () => 'changed' })).rejects.toThrow(/same project/);
    expect(uploads).toEqual([]);
  });
  it('refuses a concurrent owner of the same eval state', async () => {
    const statePath = join(directory, 'locked.json');
    writeFileSync(`${statePath}.lock`, '{}', { mode: 0o600 });
    let executions = 0;
    await expect(runEval({ grading, dataset: datasetId!, label: 'locked', statePath,
      api: { projectId: projectId!, token: 'task-token', baseUrl }, task: () => { executions++; return 'answer'; } })).rejects.toThrow(/locked/);
    expect(executions).toBe(0);
    expect(paths).toEqual([]);
  });
});
