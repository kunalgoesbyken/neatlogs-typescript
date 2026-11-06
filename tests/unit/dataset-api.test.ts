import { afterEach, describe, expect, it, vi } from 'vitest';
import { DatasetApi, json, serializeCaptureBody, uuid } from '../../src/dataset/api.js';

const key = '53f5bc6f-2cc5-4e03-8c26-d0df0283845e';
const options = { token: ' control-plane-token ', projectId: key, baseUrl: 'https://api.example.test' };
afterEach(() => vi.unstubAllGlobals());

describe('authenticated dataset API transport', () => {
  it('freezes writes before retrying and uses separate token/project headers', async () => {
    const calls: RequestInit[] = [];
    vi.stubGlobal('fetch', vi.fn(async (_url, init: RequestInit) => {
      calls.push(init);
      if (calls.length === 1) return new Response('', { status: 503 });
      return Response.json({ success: true, data: { datasetId: key } });
    }));
    const body = { name: 'examples', cases: [] };
    const request = new DatasetApi(options).request('POST', '/datasets', body, key);
    body.name = 'mutated';
    await expect(request).resolves.toEqual({ datasetId: key });
    expect(calls).toHaveLength(2);
    expect(calls[0].body).toBe(calls[1].body);
    expect(JSON.parse(String(calls[0].body))).toEqual({ name: 'examples', cases: [] });
    expect(calls[0].headers).toMatchObject({ Authorization: 'Bearer control-plane-token', 'x-project-id': key, 'Idempotency-Key': key });
    expect(calls[0].redirect).toBe('error');
  });

  it('negotiates catalog cursors on first and older pages while accepting an unchanged legacy response', async () => {
    const fetch = vi.fn(async (_url: string, _init: RequestInit) => Response.json({ success: true, data: { datasets: [] } }));
    vi.stubGlobal('fetch', fetch);
    const api = new DatasetApi(options);
    await expect(api.request('GET', '/datasets')).resolves.toEqual({ datasets: [] });
    await api.request('GET', `/datasets?before=${key}`);
    await api.request('GET', `/datasets/${key}`);
    await api.request('POST', '/datasets', { name: 'test' }, key);
    expect(fetch.mock.calls.map(([, init]) => new Headers(init.headers).get('Prefer'))).toEqual(['pagination=cursor', 'pagination=cursor', null, null]);
    for (const [, init] of fetch.mock.calls) expect(init.headers).toMatchObject({ Authorization: 'Bearer control-plane-token', 'x-project-id': key });
  });

  it.each(['http://api.example.test', 'https://user:pass@api.example.test', 'https://api.example.test/path', 'https://api.example.test?secret=x'])('rejects unsafe token destination %s', (baseUrl) => {
    expect(() => new DatasetApi({ ...options, baseUrl })).toThrow();
  });

  it('allows loopback HTTP and rejects missing auth or invalid project IDs', () => {
    expect(() => new DatasetApi({ ...options, baseUrl: 'http://127.0.0.1:3056' })).not.toThrow();
    expect(() => new DatasetApi({ ...options, token: '' })).toThrow('token');
    expect(() => new DatasetApi({ ...options, projectId: 'wrong' })).toThrow('UUID');
    expect(uuid(key.toUpperCase(), 'id')).toBe(key);
  });

  it('does not retry auth/validation failures or accept malformed success responses', async () => {
    const fetch = vi.fn(async () => new Response('', { status: 403 }));
    vi.stubGlobal('fetch', fetch);
    await expect(new DatasetApi(options).request('POST', '/datasets', {}, key)).rejects.toMatchObject({ statusCode: 403 });
    expect(fetch).toHaveBeenCalledOnce();
    fetch.mockImplementation(async () => Response.json({ data: {} }));
    await expect(new DatasetApi(options).request('GET', '/datasets')).rejects.toThrow('envelope');
  });

  it('bounds request timeout', async () => {
    vi.stubGlobal('fetch', vi.fn((_url, init: RequestInit) => new Promise((_resolve, reject) => {
      init.signal?.addEventListener('abort', () => reject(new Error('aborted')));
    })));
    await expect(new DatasetApi({ ...options, requestTimeoutMillis: 5 }).request('GET', '/datasets')).rejects.toThrow('timed out');
  });
});

describe('lossless bounded dataset JSON', () => {
  it.each([undefined, NaN, Infinity, 1n, () => 1, new Date(), { missing: undefined }, [undefined], new Array(1)])('rejects values JSON would change: %#', (value) => {
    expect(() => json(value)).toThrow();
  });
  it('rejects cycles and hidden properties but permits repeated objects', () => {
    const cycle: { self?: unknown } = {};
    cycle.self = cycle;
    expect(() => json(cycle)).toThrow();
    expect(() => json({ [Symbol('secret')]: 1 })).toThrow();
    const repeated = { hello: 'world' };
    expect(json([repeated, repeated])).toEqual([repeated, repeated]);
  });
  it('checks bytes without truncation', () => {
    expect(() => serializeCaptureBody({ text: '界'.repeat(22_000) })).toThrow('64 KiB');
    expect(serializeCaptureBody({ text: 'fine' })).toBe('{"text":"fine"}');
  });
});
