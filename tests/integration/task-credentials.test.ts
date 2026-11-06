import { createServer, type Server } from 'node:http';
import { randomUUID } from 'node:crypto';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { DatasetApi } from '../../src/dataset/api.js';
import { LocalRunApi } from '../../src/replay/run-api.js';
import { taskCredentialRefresh } from '../../src/task-credentials.js';

const projectId = randomUUID(), runId = randomUUID(), caseId = randomUUID();
const newToken = `nloa_${'n'.repeat(43)}`;
let server: Server, baseUrl: string, renewals: number, status: number;
let delaySecondRejection = false;
let refreshDelay = 15;
let calls: Array<{ path: string; key: string | undefined; body: string }>;
beforeEach(async () => {
  renewals = 0; status = 200; calls = []; delaySecondRejection = false; refreshDelay = 15;
  server = createServer(async (request, response) => {
    const chunks: Buffer[] = [];
    for await (const chunk of request) chunks.push(Buffer.from(chunk));
    const body = Buffer.concat(chunks).toString();
    response.setHeader('Content-Type', 'application/json');
    if (request.url === '/refresh') {
      expect(request.headers.authorization).toBe('Bearer private-capability');
      renewals++;
      await new Promise(resolve => setTimeout(resolve, refreshDelay));
      response.end(JSON.stringify({ token: newToken }));
      return;
    }
    calls.push({ path: request.url!, key: typeof request.headers['idempotency-key'] === 'string' ? request.headers['idempotency-key'] : undefined, body });
    const code = status !== 200 ? status : request.headers.authorization === `Bearer ${newToken}` ? 200 : 401;
    if (delaySecondRejection && code === 401 && calls.length === 2) await new Promise(resolve => setTimeout(resolve, 100));
    response.writeHead(code);
    response.end(JSON.stringify(code === 200 ? { success: true, data: request.method === 'GET' ? {} : { outputId: randomUUID(), state: 'succeeded' } } : { code: 'UNAUTHORIZED' }));
  });
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('Missing test address');
  baseUrl = `http://127.0.0.1:${address.port}`;
  vi.stubEnv('NEATLOGS_TOKEN', 'old-token');
  vi.stubEnv('NEATLOGS_HOST', baseUrl); vi.stubEnv('NEATLOGS_PROJECT_ID', projectId);
  vi.stubEnv('NEATLOGS_CREDENTIAL_BRIDGE_URL', `${baseUrl}/refresh`);
  vi.stubEnv('NEATLOGS_CREDENTIAL_BRIDGE_KEY', 'private-capability');
});
afterEach(async () => { vi.unstubAllEnvs(); server.closeAllConnections(); await new Promise<void>(resolve => server.close(() => resolve())); });
function options() {
  const api = { baseUrl, projectId, token: 'old-token' };
  return { ...api, refreshAccessToken: taskCredentialRefresh(api) };
}

describe('CLI task token renewal', () => {
  it('waits for a renewal lasting longer than five seconds', async () => {
    refreshDelay = 6_000;
    await expect(new DatasetApi(options()).request('GET', '/datasets')).resolves.toEqual({});
    expect(renewals).toBe(1);
    expect(calls).toHaveLength(2);
  }, 10_000);
  it('does not rotate a new token when an older request is rejected late', async () => {
    delaySecondRejection = true;
    const api = new DatasetApi(options());
    await Promise.all([api.request('GET', '/datasets'), api.request('GET', '/datasets')]);
    expect(renewals).toBe(1);
  });
  it('shares one renewal across concurrent dataset reads', async () => {
    const api = new DatasetApi(options());
    await Promise.all([api.request('GET', '/datasets'), api.request('GET', '/datasets')]);
    expect(renewals).toBe(1);
    expect(calls).toHaveLength(4);
  });
  it('retries an output upload without changing its body or idempotency key', async () => {
    const api = new LocalRunApi(options());
    await api.taskOutput(runId, { caseId, trial: 1, output: 'saved answer' }, randomUUID());
    expect(renewals).toBe(1);
    expect(calls).toHaveLength(2);
    expect(calls[0]).toEqual(calls[1]);
  });
  it.each([401, 403])('stops on persistent HTTP %i without a refresh loop', async code => {
    status = code;
    await expect(new LocalRunApi(options()).taskOutput(runId, { caseId, trial: 1, output: 'answer' })).rejects.toMatchObject({ status: code });
    expect(renewals).toBe(code === 401 ? 1 : 0);
    expect(calls).toHaveLength(code === 401 ? 2 : 1);
  });
  it('does not supply a credential to a different project or API host', () => {
    expect(taskCredentialRefresh({ baseUrl, projectId: randomUUID(), token: 'old' })).toBeUndefined();
    expect(taskCredentialRefresh({ baseUrl, projectId, token: 'explicit-other-token' })).toBeUndefined();
    expect(taskCredentialRefresh({ baseUrl: 'https://other.example', projectId, token: 'old' })).toBeUndefined();
    vi.stubEnv('NEATLOGS_CREDENTIAL_BRIDGE_URL', 'https://external.example/refresh');
    expect(() => options()).toThrow('Invalid CLI credential bridge');
    expect(renewals).toBe(0);
  });
});
