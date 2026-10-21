import { createServer, type Server } from 'node:http';
import { randomUUID } from 'node:crypto';
import { mkdtempSync, rmSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { Dataset, runEval } from '../../src/evals.js';

const [projectId, datasetId, versionId, caseId, revisionId, runId] = Array.from({ length: 6 }, () => randomUUID());
let server: Server;
let baseUrl: string;
let directory: string;
let failUpload = false;
let verdict = 'pass';
let uploads: Record<string, unknown>[];
let paths: string[];
beforeEach(async () => {
  uploads = []; paths = []; failUpload = false; verdict = 'pass'; directory = mkdtempSync(join(tmpdir(), 'neatlogs-eval-'));
  server = createServer(async (request, response) => {
    const path = request.url ?? ''; paths.push(path);
    const chunks = []; for await (const chunk of request) chunks.push(Buffer.from(chunk));
    const body = chunks.length ? JSON.parse(Buffer.concat(chunks).toString()) : {};
    let data: unknown, status = 200;
    if (request.method === 'GET') {
      if (path.includes('/comparison')) data = { state: 'ready', coverage: { complete: true } };
      else if (path.includes(`/cases/${caseId}`)) data = { case: { caseId, revisionId, input: 'hello', expected: { value: 'hello' }, metadata: { language: 'en' } } };
      else if (path.includes('/cases?')) data = { cases: [{ caseId }], total: 1 };
      else data = { dataset: { caseCount: 1 }, version: { id: versionId } };
    } else {
      expect(request.headers.authorization).toBe('Bearer task-token'); expect(request.headers['x-project-id']).toBe(projectId); expect(request.headers['idempotency-key']).toBeTruthy();
      if (path.endsWith('/outputs')) { if (failUpload) status = 503; else uploads.push(body); data = { outputId: randomUUID(), state: 'succeeded' }; }
      else if (path.endsWith('/gate')) data = { verdict };
      else data = { runId, state: path.endsWith('/complete') ? 'completed' : 'queued', deadlineAt: '2030-01-01T00:00:00Z', itemCount: 2, completedCount: 2, failedCount: 0, pendingCount: 0, gradingHandoffState: 'dispatched' };
      if (!path.endsWith('/complete') && !path.endsWith('/gate') && !path.endsWith('/outputs')) {
        const { completedCount: _completed, failedCount: _failed, pendingCount: _pending, gradingHandoffState: _handoff, ...receipt } = data as Record<string, unknown>; data = receipt;
      }
    }
    response.writeHead(status, { 'Content-Type': 'application/json' }); response.end(JSON.stringify({ success: status < 400, data }));
  });
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  const address = server.address(); if (!address || typeof address === 'string') throw new Error('No local test address'); baseUrl = `http://127.0.0.1:${address.port}`;
});
afterEach(async () => { await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve())); rmSync(directory, { recursive: true, force: true }); });

describe('task evaluation over the public HTTP contract', () => {
  it('runs each trial, carries context and maps failed gates to exit1', async () => {
    verdict = 'fail'; const seen: number[] = [];
    const result = await runEval({ dataset: new Dataset(datasetId!, { tag: 'golden-v3' }), label: 'candidate', params: { model: 'small' }, trials: 2, gates: ['new_failures <= 0'], api: { projectId: projectId!, token: 'task-token', baseUrl }, task: (_input, context) => { seen.push(context.trial); expect(context.param('model')).toBe('small'); context.metrics.costUsd = 0.001; return { answer: 'hello' }; } });
    expect(result.exitCode).toBe(1); expect(seen).toEqual([1, 2]); expect(uploads).toHaveLength(2); expect(paths.some(path => path.includes('?tag=golden-v3'))).toBe(true);
  });
  it('resumes a saved upload without repeating a completed task', async () => {
    const seen: number[] = [], statePath = join(directory, 'run.json');
    const options = { dataset: datasetId!, label: 'resume', trials: 2, statePath, api: { projectId: projectId!, token: 'task-token', baseUrl }, task: (_input: unknown, context: { trial: number }) => { seen.push(context.trial); return 'saved output'; } };
    failUpload = true; await expect(runEval(options)).rejects.toThrow(/HTTP 503/); expect(seen).toEqual([1]); expect(statSync(statePath).mode & 0o077).toBe(0);
    failUpload = false; const result = await runEval({ ...options, resume: true }); expect(result.exitCode).toBe(0); expect(seen).toEqual([1, 2]); expect(uploads).toHaveLength(2);
  });
  it('retains task errors and returns incomplete', async () => {
    const result = await runEval({ dataset: datasetId!, label: 'error', api: { projectId: projectId!, token: 'task-token', baseUrl }, task: () => { throw new Error('task failed'); } });
    expect(result.exitCode).toBe(4); expect(result.errors).toBe(1); expect(uploads[0]?.error).toEqual({ code: 'Error', message: 'task failed' });
  });
  it('rejects changed task code on resume', async () => {
    const options = { dataset: datasetId!, label: 'resume', statePath: join(directory, 'run.json'), api: { projectId: projectId!, token: 'task-token', baseUrl }, task: () => 'original' };
    failUpload = true; await expect(runEval(options)).rejects.toThrow(/HTTP 503/);
    failUpload = false; await expect(runEval({ ...options, resume: true, task: () => 'changed' })).rejects.toThrow(/same project/);
    expect(uploads).toEqual([]);
  });
});
