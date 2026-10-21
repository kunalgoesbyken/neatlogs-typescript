import { randomUUID } from 'node:crypto';
import { execFile } from 'node:child_process';
import { createServer, type Server, type RequestListener } from 'node:http';
import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { promisify } from 'node:util';
import { afterEach, describe, expect, it } from 'vitest';
import { DatasetApi } from '../../src/dataset/api.js';
import { enumerateSavedDataset, datasetSelectionDigest } from '../../src/replay/all-selection.js';
import { agentConfigDigest, createAllReplayState, readAllReplayState, saveAllReplayState, withAllReplayLock, type AllReplayConfiguration } from '../../src/replay/all-state.js';

const exec = promisify(execFile), servers: Server[] = [], directories: string[] = [];
const ids = { project: randomUUID(), dataset: randomUUID(), version: randomUUID(), form: randomUUID(), evaluator: randomUUID() };
const members = (count: number) => Array.from({ length: count }, (_, position) => ({ caseId: randomUUID(), revisionId: randomUUID(), position }));
async function serve(handler: RequestListener) {
  const server = createServer(handler); servers.push(server);
  await new Promise<void>(done => server.listen(0, '127.0.0.1', done));
  const address = server.address(); if (!address || typeof address === 'string') throw new Error('Missing port.');
  return `http://127.0.0.1:${address.port}`;
}
const send = (response: Parameters<RequestListener>[1], data: unknown) => { response.setHeader('content-type', 'application/json'); response.end(JSON.stringify({ success: true, data })); };
const options = (baseUrl: string) => ({ baseUrl, token: 'private-test-token', projectId: ids.project });
async function workspace() { const path = await mkdtemp(join(tmpdir(), 'neatlogs-all-test-')); directories.push(path); return path; }
afterEach(async () => {
  await Promise.all(servers.splice(0).map(server => new Promise<void>((done, reject) => server.close(error => error ? reject(error) : done()))));
  await Promise.all(directories.splice(0).map(path => rm(path, { recursive: true, force: true })));
});

describe('entire saved dataset replay', () => {
  it('enumerates all201 immutable members in three bounded HTTP pages without latest-version drift', async () => {
    const cases = members(201), offsets: number[] = [];
    const baseUrl = await serve((request, response) => {
      const url = new URL(request.url!, 'http://local');
      expect(url.searchParams.get('versionId')).toBe(ids.version);
      expect(request.headers.authorization).toBe('Bearer private-test-token');
      if (!url.pathname.endsWith('/cases')) return send(response, { dataset: { id: ids.dataset, caseCount: cases.length, latestVersionId: randomUUID() }, version: { id: ids.version } });
      const offset = Number(url.searchParams.get('offset')); offsets.push(offset);
      expect(url.searchParams.get('limit')).toBe('100');
      send(response, { total: cases.length, cases: cases.slice(offset, offset + 100) });
    });
    expect(await enumerateSavedDataset(new DatasetApi(options(baseUrl)), ids.dataset, ids.version)).toEqual(cases);
    expect(offsets).toEqual([0, 100, 200]);
    expect(datasetSelectionDigest({ projectId: ids.project, datasetId: ids.dataset, versionId: ids.version, cases })).not.toBe(datasetSelectionDigest({ projectId: randomUUID(), datasetId: ids.dataset, versionId: ids.version, cases }));
  });

  it.each(['wrong_version', 'duplicate', 'changed_total', 'missing_page'] as const)('rejects %s before executing any agent', async failure => {
    const cases = members(101);
    const baseUrl = await serve((request, response) => {
      const url = new URL(request.url!, 'http://local');
      if (!url.pathname.endsWith('/cases')) return send(response, { dataset: { id: ids.dataset, caseCount: 101 }, version: { id: failure === 'wrong_version' ? randomUUID() : ids.version } });
      const offset = Number(url.searchParams.get('offset'));
      send(response, { total: failure === 'changed_total' ? 100 : 101, cases: failure === 'missing_page' ? [] : failure === 'duplicate' && offset ? [{ ...cases[0], position: 100 }] : cases.slice(offset, offset + 100) });
    });
    await expect(enumerateSavedDataset(new DatasetApi(options(baseUrl)), ids.dataset, ids.version)).rejects.toThrow();
  });

  it('preserves private immutable state and refuses concurrent owners and selection tampering', async () => {
    const directory = await workspace(), path = join(directory, 'state.json');
    const configuration: AllReplayConfiguration = { projectId: ids.project, datasetId: ids.dataset, versionId: ids.version, baseUrl: 'http://127.0.0.1:1', cwd: process.cwd(), agentFile: '/agent.yaml', agentDigest: agentConfigDigest({ command: ['node', 'agent.mjs'], timeout_seconds: 1 }), label: 'All', trials: 1, batchSize: 2, deadlineSeconds: 60, replay: { mode: 'tools_only' }, params: {}, grading: null };
    const state = createAllReplayState(configuration, members(3));
    await withAllReplayLock(path, async () => {
      await saveAllReplayState(path, state, true);
      await expect(withAllReplayLock(path, async () => undefined)).rejects.toThrow('locked');
    });
    expect((await stat(path)).mode & 0o777).toBe(0o600);
    expect(await readAllReplayState(path)).toEqual(state);
    state.cases[0]!.revisionId = randomUUID();
    await saveAllReplayState(path, state);
    await expect(readAllReplayState(path)).rejects.toThrow('changed');
  });

  it('runs real child processes in bounded batches and resumes upload without repeating a completed command', async () => {
    const directory = await workspace(), configFile = join(directory, 'agent.yaml'), agentFile = join(directory, 'agent.mts'), stateFile = join(directory, 'state.json'), logFile = join(directory, 'calls.log');
    await writeFile(agentFile, `import { appendFile } from 'node:fs/promises';
import { init, shutdown } from ${JSON.stringify(resolve('src/init.ts'))};
import { span } from ${JSON.stringify(resolve('src/decorators/index.ts'))};
await init({ disableExport: true });
let raw='';for await(const chunk of process.stdin)raw+=chunk; const input=JSON.parse(raw);
const lookup=span({kind:'TOOL',name:'lookup',replay:true},async()=>{throw new Error('Live tool forbidden');});
await lookup(input);await appendFile(${JSON.stringify(logFile)},input.customer+'\\n');await shutdown();
if(process.env.FAIL_AGENT==='true'&&input.customer==='2')process.exitCode=7;`);
    const agent = { command: [process.execPath, '--import', 'tsx', agentFile], timeout_seconds: 20 };
    await writeFile(configFile, JSON.stringify(agent));
    const cases = members(3), registrations: Array<{ key: string; body: Record<string, unknown> }> = [], uploads: Array<{ key: string; body: Record<string, unknown> }> = [];
    let failUploads = true;
    const baseUrl = await serve(async (request, response) => {
      const url = new URL(request.url!, 'http://local');
      let raw = ''; for await (const chunk of request) raw += chunk;
      const body = raw ? JSON.parse(raw) : {};
      if (request.method === 'GET') {
        if (url.pathname.startsWith('/api/v1/public/runs/')) return send(response, { runId: url.pathname.split('/').at(-1), state: 'running', deadlineAt: new Date(Date.now() + 900_000).toISOString(), itemCount: 2, completedCount: 0, failedCount: 0, pendingCount: 2, gradingHandoffState: 'pending' });
        expect(url.searchParams.get('versionId')).toBe(ids.version);
        if (url.pathname.endsWith('/cases')) return send(response, { total: cases.length, cases });
        const item = cases.find(item => url.pathname.endsWith(item.caseId));
        if (item) return send(response, { position: item.position, case: { ...item, input: { customer: String(item.position) }, source: { traceId: `source-${item.position}` }, context: { spans: [{ span_id: 'tool', span_type: 'TOOL', tool_name: 'lookup', input_value: JSON.stringify({ customer: String(item.position) }), output_value: '{"ok":true}' }] } } });
        return send(response, { dataset: { id: ids.dataset, caseCount: cases.length }, version: { id: ids.version } });
      }
      if (url.pathname.endsWith('/runs')) {
        registrations.push({ key: String(request.headers['idempotency-key']), body });
        return send(response, { runId: randomUUID(), state: 'queued', deadlineAt: new Date(Date.now() + 900_000).toISOString(), itemCount: body.caseIds.length });
      }
      uploads.push({ key: String(request.headers['idempotency-key']), body });
      if (failUploads) { response.statusCode = 503; response.end('{}'); return; }
      send(response, { runId: url.pathname.split('/').at(-2), itemId: randomUUID(), outputId: randomUUID(), traceId: body.traceId, state: 'succeeded' });
    });
    const env = { ...process.env, NEATLOGS_TOKEN: 'private-test-token', NEATLOGS_PROJECT_ID: ids.project, NEATLOGS_HOST: baseUrl };
    const invoke = (args: string[], environment: NodeJS.ProcessEnv = {}) => exec(process.execPath, ['--import', 'tsx', resolve('src/cli.ts'), 'simulate', 'local', ...args], { cwd: process.cwd(), env: { ...env, ...environment }, timeout: 40_000 });
    await expect(invoke(['--dataset', ids.dataset, '--version', ids.version, '--all', '--state', stateFile, '--batch-size', '2', '--agent', configFile, '--upload', '--form', ids.form, '--evaluator', ids.evaluator, '--json'])).rejects.toMatchObject({ code: 1 });
    expect((await readFile(logFile, 'utf8')).trim().split('\n')).toEqual(['0']);
    const saved = await readAllReplayState(stateFile);
    expect(saved.batches[0]!.slots[0]!.progress.kind).toBe('upload_pending');
    expect(saved.batches[0]!.slots[0]!.execution).toMatchObject({ sourceTraceId: 'source-0', exitCode: 0,
      status: { sourceTraceId: 'source-0', mode: 'tools_only', hits: 1, liveCalls: 0, firstMiss: null } });
    expect(await readFile(stateFile, 'utf8')).not.toContain('private-test-token');
    expect(await readFile(stateFile, 'utf8')).not.toContain('agent.mts');
    const firstRun = saved.batches[0]!.run!.runId;
    await expect(invoke(['--resume', stateFile, '--case', cases[0]!.caseId])).rejects.toThrow();
    await expect(invoke(['--resume', stateFile], { NEATLOGS_PROJECT_ID: randomUUID() })).rejects.toThrow();
    await expect(invoke(['--resume', stateFile], { NEATLOGS_HOST: 'http://127.0.0.1:1' })).rejects.toThrow();
    failUploads = false;
    const resumed = JSON.parse((await invoke(['--resume', stateFile, '--json'])).stdout);
    expect(resumed).toMatchObject({ exitCode: 0, completedAttempts: 3, totalCases: 3, runs: [{ batch: 1, runId: firstRun }, { batch: 2 }] });
    expect(resumed.reports[0]).toMatchObject({ caseId: cases[0]!.caseId, trial: 1, progress: 'accepted', ...saved.batches[0]!.slots[0]!.execution });
    expect(resumed.reports.map((report: { status: { hits: number } }) => report.status.hits)).toEqual([1, 1, 1]);
    expect((await readFile(logFile, 'utf8')).trim().split('\n')).toEqual(['0', '1', '2']);
    expect(registrations.map(call => call.body.caseIds)).toEqual([cases.slice(0, 2).map(item => item.caseId), [cases[2]!.caseId]]);
    expect(registrations.map(call => call.body.selection)).toEqual([0, 2].map(batchOffset => ({ kind: 'all', groupId: saved.groupId, datasetId: ids.dataset, digest: saved.selectionDigest, totalCases: 3, batchOffset, batchSize: 2 })));
    expect(uploads[0]!.key).toBe(uploads[2]!.key); expect(uploads[0]!.body).toEqual(uploads[2]!.body);
    expect(JSON.parse((await invoke(['--resume', stateFile, '--json'])).stdout).reports).toEqual(resumed.reports);
    expect((await invoke(['--resume', stateFile])).stdout).toContain('HIT 1; MISS 0; LIVE 0');
    expect((await readFile(logFile, 'utf8')).trim().split('\n')).toHaveLength(3);
    const complete = await readAllReplayState(stateFile);
    for (const field of ['sourceTraceId', 'runId', 'traceId', 'mode'] as const) {
      const changed = structuredClone(complete), slot = changed.batches[0]!.slots[0]!;
      if (field === 'sourceTraceId') slot.execution!.status!.sourceTraceId = 'other-source';
      else if (field === 'runId') slot.execution!.runId = randomUUID();
      else if (field === 'traceId') slot.execution!.status!.traceIds = ['f'.repeat(32)];
      else slot.execution!.status!.mode = 'until_span';
      await saveAllReplayState(stateFile, changed);
      await expect(readAllReplayState(stateFile)).rejects.toThrow('activity');
    }
    await saveAllReplayState(stateFile, complete);

    const localState = join(directory, 'local-state.json');
    let localResult: unknown;
    try { await invoke(['--dataset', ids.dataset, '--version', ids.version, '--all', '--state', localState, '--agent', configFile, '--json'], { FAIL_AGENT: 'true' }); }
    catch (error) { localResult = error; }
    expect(localResult).toMatchObject({ code: 1 });
    const localSaved = await readAllReplayState(localState), localSlots = localSaved.batches[0]!.slots;
    expect(localSlots.map(slot => slot.progress.kind)).toEqual(['local_complete', 'local_complete', 'failed']);
    expect(localSlots.map(slot => slot.execution?.status?.hits)).toEqual([1, 1, 1]);
    expect(localSlots.every(slot => slot.execution?.runId === null && slot.execution.status?.traceIds.length === 1)).toBe(true);
    expect(localSlots[2]!.execution?.exitCode).toBe(7);
    await expect(invoke(['--resume', localState, '--json'])).rejects.toMatchObject({ code: 1 });
    expect((await readFile(logFile, 'utf8')).trim().split('\n')).toHaveLength(6);

    const uncertain = await readAllReplayState(stateFile);
    uncertain.batches[0]!.slots[0]!.progress = { kind: 'executing' };
    uncertain.batches[0]!.slots[0]!.execution!.status = null;
    uncertain.batches[0]!.slots[0]!.execution!.exitCode = null;
    await saveAllReplayState(stateFile, uncertain);
    await expect(invoke(['--resume', stateFile, '--json'])).rejects.toMatchObject({ code: 1 });
    expect((await readFile(logFile, 'utf8')).trim().split('\n')).toHaveLength(6);
    await writeFile(configFile, JSON.stringify({ ...agent, timeout_seconds: 19 }));
    await expect(invoke(['--resume', stateFile])).rejects.toThrow();
  }, 120_000);
});
