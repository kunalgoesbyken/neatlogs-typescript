import { context } from '@opentelemetry/api';
import { suppressTracing } from '@opentelemetry/core';

export type DatasetJson = null | boolean | number | string | DatasetJson[] | { [key: string]: DatasetJson };

export interface DatasetApiOptions {
  token: string;
  projectId: string;
  baseUrl: string;
  requestTimeoutMillis?: number;
}

export class DatasetCaptureError extends Error {
  constructor(message: string, readonly statusCode?: number) {
    super(message);
    this.name = 'DatasetCaptureError';
  }
}

export function record(value: unknown): Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new DatasetCaptureError('Invalid dataset API response.');
  }
  return Object.fromEntries(Object.entries(value));
}

export function uuid(value: unknown, field: string): string {
  if (typeof value !== 'string' || !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(value)) {
    throw new DatasetCaptureError(`${field} must be a UUID.`);
  }
  return value.toLowerCase();
}

export function positive(value: number, field: string): number {
  if (!Number.isSafeInteger(value) || value <= 0 || value > 2_147_483_647) {
    throw new DatasetCaptureError(`${field} must be a positive integer below 2147483648.`);
  }
  return value;
}

/** Reject values JSON.stringify would silently change or omit. */
export function json(value: unknown, ancestors = new Set<object>()): DatasetJson {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value !== 'object' || value === null || ancestors.has(value)) {
    throw new DatasetCaptureError('Dataset values must contain finite, acyclic JSON only.');
  }
  ancestors.add(value);
  try {
    if (Array.isArray(value)) return Array.from(value, (item) => json(item, ancestors));
    if (Object.getPrototypeOf(value) !== Object.prototype && Object.getPrototypeOf(value) !== null) {
      throw new DatasetCaptureError('Dataset values must contain plain JSON objects.');
    }
    if (Reflect.ownKeys(value).length !== Object.keys(value).length) {
      throw new DatasetCaptureError('Dataset values cannot contain hidden or symbol properties.');
    }
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, json(item, ancestors)]));
  } finally {
    ancestors.delete(value);
  }
}

export function serializeCaptureBody(body: unknown): string {
  const serialized = JSON.stringify(json(body));
  if (Buffer.byteLength(serialized) > 65_536) {
    throw new DatasetCaptureError('Dataset request exceeds 64 KiB; no cases or traces were truncated.');
  }
  return serialized;
}

async function boundedResponse(response: Response): Promise<unknown> {
  const reader = response.body?.getReader();
  if (!reader) throw new DatasetCaptureError('Empty dataset API response.');
  const chunks: Uint8Array[] = []; let bytes = 0;
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      bytes += value.length;
      if (bytes > 17 * 1024 * 1024) throw new DatasetCaptureError('Dataset API response exceeds 17 MiB.');
      chunks.push(value);
    }
    return JSON.parse(Buffer.concat(chunks).toString('utf8'));
  } finally { await reader.cancel(); }
}

export class DatasetApi {
  private readonly baseUrl: string;
  private readonly token: string;
  private readonly projectId: string;
  private readonly timeout: number;

  constructor(options: DatasetApiOptions) {
    if (typeof options.token !== 'string' || !options.token.trim()) {
      throw new DatasetCaptureError('An explicit dataset API token is required; ingestion API keys are not reused.');
    }
    this.token = options.token.trim();
    this.projectId = uuid(options.projectId, 'projectId');
    let url: URL;
    try { url = new URL(options.baseUrl); }
    catch { throw new DatasetCaptureError('baseUrl must be an explicit HTTP(S) API origin.'); }
    const loopback = ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname);
    if ((url.protocol !== 'https:' && !(url.protocol === 'http:' && loopback)) || url.username || url.password || url.search || url.hash || url.pathname !== '/') {
      throw new DatasetCaptureError('baseUrl must be an HTTPS API origin; HTTP is allowed only on loopback.');
    }
    this.baseUrl = url.origin;
    this.timeout = positive(options.requestTimeoutMillis ?? 30_000, 'requestTimeoutMillis');
  }

  request(method: 'GET' | 'POST', path: string, body?: unknown, idempotencyKey?: string, timeoutMillis = this.timeout): Promise<unknown> {
    return this.requestSerialized(method, path, body === undefined ? undefined : serializeCaptureBody(body), idempotencyKey, timeoutMillis);
  }

  async requestSerialized(method: 'GET' | 'POST', path: string, serialized?: string, idempotencyKey?: string, timeoutMillis = this.timeout): Promise<unknown> {
    return (await this.requestSerializedWithMetadata(method, path, serialized, idempotencyKey, timeoutMillis)).data;
  }

  async requestSerializedWithMetadata(method: 'GET' | 'POST', path: string, serialized?: string, idempotencyKey?: string, timeoutMillis = this.timeout): Promise<{ data: unknown; idempotencyReplayed: boolean }> {
    if (!/^\/(datasets|dataset-capture-jobs)(\/|\?|$)/.test(path) || path.includes('..')) {
      throw new DatasetCaptureError('Invalid dataset API path.');
    }
    if (method === 'POST' && !idempotencyKey) throw new DatasetCaptureError('Dataset writes require an idempotency key.');
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), Math.min(this.timeout, positive(timeoutMillis, 'timeoutMillis')));
    try {
      const headers = {
        Authorization: `Bearer ${this.token}`,
        'x-project-id': this.projectId,
        ...(serialized !== undefined ? { 'Content-Type': 'application/json' } : {}),
        ...(idempotencyKey ? { 'Idempotency-Key': uuid(idempotencyKey, 'Idempotency-Key') } : {}),
      };
      for (let attempt = 0; ; attempt += 1) {
        let response: Response;
        try {
          response = await context.with(suppressTracing(context.active()), () => fetch(`${this.baseUrl}/api/v1/public${path}`, {
            method, signal: controller.signal, headers, body: serialized, redirect: 'error',
          }));
        } catch (error) {
          if (attempt === 0 && !controller.signal.aborted) continue;
          throw error;
        }
        if (attempt === 0 && [500, 502, 503, 504].includes(response.status) && !controller.signal.aborted) {
          await response.body?.cancel();
          continue;
        }
        if (!response.ok) {
          await response.body?.cancel();
          throw new DatasetCaptureError(`Dataset ${method} failed with HTTP ${response.status}.`, response.status);
        }
        const envelope = record(await boundedResponse(response));
        if (envelope.success !== true || !('data' in envelope)) throw new DatasetCaptureError('Invalid dataset API envelope.');
        return {
          data: envelope.data,
          idempotencyReplayed: response.headers.get('Idempotency-Replayed') === 'true',
        };
      }
    } catch (error) {
      if (error instanceof DatasetCaptureError) throw error;
      throw new DatasetCaptureError(controller.signal.aborted ? 'Dataset API request timed out.' : 'Dataset API request failed.');
    } finally {
      clearTimeout(timer);
    }
  }
}
