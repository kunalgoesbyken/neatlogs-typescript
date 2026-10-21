import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { execFile } from 'node:child_process';
import { createServer, type Server, type RequestListener } from 'node:http';
import { LocalRunApi } from '../../src/replay/run-api.js';
import { promisify } from 'node:util';
import { afterEach, describe, expect, it } from 'vitest';
import { parseAgentConfig, runLocalCommand } from '../../src/replay/local-command.js';

const directories: string[] = [];
const servers: Server[] = [];
const ids = { project: '11111111-1111-4111-8111-111111111111', dataset: '22222222-2222-4222-8222-222222222222', version: '33333333-3333-4333-8333-333333333333', case: '44444444-4444-4444-8444-444444444444', run: '55555555-5555-4555-8555-555555555555', form: '66666666-6666-4666-8666-666666666666', evaluator: '77777777-7777-4777-8777-777777777777', output: '88888888-8888-4888-8888-888888888888' };
async function serve(handler: RequestListener) {
  const server = createServer(handler); servers.push(server);
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('No server port.');
  return `http://127.0.0.1:${address.port}`;
}
const recording = { version: 1, sourceTraceId: 'source-1', entries: [
  { spanId: 'tool-1', kind: 'tool', name: 'lookup', input: { customer: '1' }, output: { balance: 40 } },
] };
const exec = promisify(execFile);
afterEach(async () => {
  await Promise.all(servers.splice(0).map(server => new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()))));
  await Promise.all(directories.splice(0).map(path => rm(path, { recursive: true, force: true })));
});
async function workspace() {
  const path = await mkdtemp(join(tmpdir(), 'neatlogs-local-test-')); directories.push(path); return path;
}
async function agent(path: string, allowLive = false): Promise<string[]> {
  const file = join(path, 'agent.mts');
  await writeFile(file, `
import { init, shutdown } from ${JSON.stringify(resolve('src/init.ts'))};
import { span } from ${JSON.stringify(resolve('src/decorators/index.ts'))};
await init({ disableExport: true });
let text = ''; for await (const chunk of process.stdin) text += chunk;
const input = JSON.parse(text);
const lookup = span({ kind: 'TOOL', name: 'lookup', replay: true }, async () => { ${allowLive ? 'return { balance: 50 };' : "throw new Error('Real tool must not run');"} });
console.log(JSON.stringify({ result: await lookup(input), params: JSON.parse(process.env.NEATLOGS_REPLAY_PARAMS || '{}') }));
await shutdown();
`);
  return [process.execPath, '--import', 'tsx', file];
}

describe('local replay command', () => {
  it('runs a real child process with replay input and private temporary files', async () => {
    const path = await workspace();
    const spanLog = join(path, 'spans.jsonl');
    const result = await runLocalCommand({ recording, replay: { mode: 'tools_only', run: { runId: ids.run, caseId: ids.case, trial: 1 } }, input: { customer: '1' },
      cwd: process.cwd(), env: { ...process.env, NEATLOGS_LOG_SPANS: 'true', NEATLOGS_LOG_SPANS_FILE: spanLog }, params: { model: 'cheap' }, config: { command: await agent(path), timeout_seconds: 20 } });
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain('"balance":40');
    expect(result.stdout).toContain('"model":"cheap"');
    expect(result.status).toMatchObject({ hits: 1, liveCalls: 0, sourceTraceId: 'source-1', switchedToLive: false });
    expect(result.status.traceIds).toHaveLength(1);
    const spans = (await readFile(spanLog, 'utf8')).trim().split('\n').map(line => JSON.parse(line));
    expect(spans.find(item => item.name === 'lookup')?.attributes).toMatchObject({
      'neatlogs.origin': 'simulation', 'neatlogs.replay.run_id': ids.run,
      'neatlogs.replay.case_id': ids.case, 'neatlogs.replay.trial': 1,
      'neatlogs.replay.outcome': 'HIT',
    });
  }, 30_000);

  it('executes the actual CLI from a frozen case and agent YAML', async () => {
    const path = await workspace();
    const file = join(path, 'case.json'), config = join(path, 'agent.yaml');
    await writeFile(file, JSON.stringify({ input: { message: { customer: '1' } }, source: { kind: 'trace', traceId: 'source-1' }, context: { spans: [
      { span_id: 'tool-1', span_type: 'TOOL', tool_name: 'lookup', input_value: JSON.stringify({ customer: '1' }), output_value: JSON.stringify({ balance: 40 }) },
    ] } }));
    await writeFile(config, JSON.stringify({ command: await agent(path), input_from_trace: { customer: 'root.input.message.customer' }, timeout_seconds: 20 }));
    const result = await exec(process.execPath, ['--import', 'tsx', resolve('src/cli.ts'), 'simulate', 'local', '--recording', file, '--agent', config, '--param', 'model=cheap', '--json'], { cwd: process.cwd(), timeout: 30000 });
    const report = JSON.parse(result.stdout);
    expect(report).toMatchObject({ exitCode: 0, uploaded: false, skippedTools: 0, status: { hits: 1, liveCalls: 0 } });
    expect(report.stdout).toContain('"balance":40');
    expect(await readFile(file, 'utf8')).toContain('source-1');
  }, 30_000);

  it('rejects bad checkpoints and unresolved input before launching code', async () => {
    const config = { command: ['does-not-exist'], timeout_seconds: 1 };
    await expect(runLocalCommand({ recording, replay: { mode: 'until_span', untilSpan: 'missing' }, input: null, config, cwd: process.cwd() })).rejects.toThrow('checkpoint');
    await expect(runLocalCommand({ recording, replay: { mode: 'tools_only' }, input: {}, config: { ...config, input_from_trace: { input: '__proto__.secret' } }, cwd: process.cwd() })).rejects.toThrow('mapping');
    expect(() => parseAgentConfig('command: rm -rf /')).toThrow();
    expect(() => parseAgentConfig('command: [node]\ncommand: [python]')).toThrow();
  });

  it('stops timed-out processes and does not report unsupported children as replayed', async () => {
    const base = { recording, replay: { mode: 'tools_only' as const }, input: null, cwd: process.cwd() };
    await expect(runLocalCommand({ ...base, config: { command: [process.execPath, '-e', 'setTimeout(()=>{},10000)'], timeout_seconds: 1 } })).rejects.toThrow('time limit');
    await expect(runLocalCommand({ ...base, config: { command: [process.execPath, '-e', 'process.stdin.resume()'], timeout_seconds: 5 } })).rejects.toThrow('did not report');
  });
  it.each(['tools_only', 'until_span'] as const)('uploads two actual child-process attempts with immutable %s settings and retries delivery with one stable receipt key', async (mode) => {
    const path = await workspace(), config = join(path, 'agent.yaml');
    await writeFile(config, JSON.stringify({ command: await agent(path, mode === 'until_span'), timeout_seconds: 20 }));
    const calls: { path: string; key: string | undefined; body: Record<string, unknown> }[] = [];
    const baseUrl = await serve(async (request, response) => {
      let raw = ''; for await (const chunk of request) raw += chunk;
      const body = raw ? JSON.parse(raw) : {}, key = request.headers['idempotency-key'] as string | undefined;
      calls.push({ path: request.url!, key, body });
      expect(request.headers.authorization).toBe('Bearer test-token');
      expect(request.headers['x-project-id']).toBe(ids.project);
      response.setHeader('content-type', 'application/json');
      if (request.method === 'GET') {
        response.end(JSON.stringify({ success: true, data: { case: { caseId: ids.case, input: { customer: '1' }, source: { traceId: 'source-1' }, context: { spans: [
          { span_id: 'tool-1', span_type: 'TOOL', tool_name: 'lookup', input_value: '{"customer":"1"}', output_value: '{"balance":40}' },
        ] } } } })); return;
      }
      if (request.url === '/api/v1/public/runs') {
        response.statusCode = 201;
        response.end(JSON.stringify({ success: true, data: { runId: ids.run, state: 'queued', deadlineAt: new Date(Date.now() + 900_000).toISOString(), itemCount: 2 } })); return;
      }
      if (calls.filter(call => call.path.endsWith('/outputs')).length === 1) {
        response.statusCode = 409; response.end(JSON.stringify({ success: false, code: 'TRACE_NOT_READY' })); return;
      }
      response.end(JSON.stringify({ success: true, data: { runId: ids.run, itemId: ids.case, outputId: ids.output, traceId: body.traceId, state: 'succeeded' } }));
    });
    const output = await exec(process.execPath, ['--import', 'tsx', resolve('src/cli.ts'), 'simulate', 'local', '--dataset', ids.dataset,
      '--version', ids.version, '--case', ids.case, '--agent', config, '--trials', '2', '--upload', '--form', ids.form, '--evaluator', ids.evaluator,
      '--param', 'temperature=0.2', '--param', 'fallback=false', ...(mode === 'until_span' ? ['--mode', mode, '--until-span', 'tool-1'] : []), '--json'], {
      cwd: process.cwd(), timeout: 30000, env: { ...process.env, NEATLOGS_TOKEN: 'test-token', NEATLOGS_PROJECT_ID: ids.project, NEATLOGS_HOST: baseUrl },
    });
    const report = JSON.parse(output.stdout);
    expect(report).toMatchObject({ runId: ids.run, exitCode: 0, reports: [{ uploaded: true, trial: 1 }, { uploaded: true, trial: 2 }] });
    const registration = calls.find(call => call.path === '/api/v1/public/runs')!;
    expect(registration.body).toMatchObject({ datasetVersionId: ids.version, caseIds: [ids.case], trials: 2, params: { temperature: 0.2, fallback: false }, replay: mode === 'until_span' ? { mode, untilSpan: 'tool-1' } : { mode } });
    expect(report.reports.every((entry: { result: { status: { mode: string; reason: string | null } } }) => entry.result.status.mode === mode && entry.result.status.reason === (mode === 'until_span' ? 'checkpoint' : null))).toBe(true);
    const uploads = calls.filter(call => call.path.endsWith('/outputs'));
    expect(uploads).toHaveLength(3);
    expect(uploads[0]!.key).toBe(uploads[1]!.key);
    expect(uploads[0]!.body).toEqual(uploads[1]!.body);
    expect(uploads[2]!.key).not.toBe(uploads[0]!.key);
    expect(uploads[2]!.body.traceId).not.toBe(uploads[0]!.body.traceId);
    expect(uploads[0]!.body.traceId).toMatch(/^[a-f0-9]{32}$/);
  }, 30_000);

  it('does not retry permanent conflicts or accept a receipt for another trace', async () => {
    let count = 0;
    const baseUrl = await serve((_request, response) => {
      count++; response.statusCode = 409;
      response.end(JSON.stringify({ success: false, error: { code: 'SLOT_ALREADY_ACCEPTED' } }));
    });
    const api = new LocalRunApi({ baseUrl, token: 'token', projectId: ids.project });
    await expect(api.upload(ids.run, { caseId: ids.case, trial: 1, traceId: 'a'.repeat(32) })).rejects.toMatchObject({ status: 409, code: 'SLOT_ALREADY_ACCEPTED' });
    expect(count).toBe(1);
    const other = await serve((_request, response) => response.end(JSON.stringify({ success: true, data: { runId: ids.run, itemId: ids.case, outputId: ids.output, traceId: 'b'.repeat(32), state: 'succeeded' } })));
    await expect(new LocalRunApi({ baseUrl: other, token: 'token', projectId: ids.project }).upload(ids.run, { caseId: ids.case, trial: 1, traceId: 'a'.repeat(32) })).rejects.toThrow('different run or trace');
  });

});
