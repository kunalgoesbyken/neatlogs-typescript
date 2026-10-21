import { createHash } from 'node:crypto';
import type { Span } from '@opentelemetry/api';
import { z } from 'zod';
import { mediaReferences, sanitizeMediaPayload } from '../core/media.js';
import type { DatasetJson } from '../dataset/api.js';
import { isCompleteProviderResponse, replayProviderOperationSchema, type ReplayProviderOperation } from './provider.js';

export const MODEL_CAPTURE_ATTRIBUTE = 'neatlogs.replay.model_capture';
const MAX_BYTES = 1024 * 1024;
const recording = z.object({
  version: z.literal(1),
  operation: replayProviderOperationSchema,
  input: z.json(),
  output: z.json(),
  digest: z.string().regex(/^[a-f0-9]{64}$/),
}).strict();
const transportKeys = /^(?:api_?key|authorization|proxyAuthorization|headers|extraHeaders|httpOptions|abortSignal|credentials|access_token|client_secret)$/i;

/** Copy actual data properties, never SDK getters, toJSON hooks, or reconstructed messages. */
function providerData(value: unknown, response = false): DatasetJson {
  let bytes = 0;
  let nodes = 0;
  const ancestors = new Set<object>();
  const visit = (item: unknown, depth: number): DatasetJson => {
    if (++nodes > 4096 || depth > 32) throw new Error('Provider capture exceeds traversal limit.');
    if (typeof item === 'string') {
      bytes += Buffer.byteLength(item);
      if (bytes > MAX_BYTES) throw new Error('Provider capture exceeds size limit.');
      return item;
    }
    if (item === null || typeof item === 'boolean' || typeof item === 'number' && Number.isFinite(item)) return item;
    if (typeof item !== 'object' || item === null || ancestors.has(item)) throw new Error('Provider capture is not JSON.');
    if ((!response || depth > 0) && !Array.isArray(item) && Object.getPrototypeOf(item) !== Object.prototype && Object.getPrototypeOf(item) !== null) throw new Error('SDK object is not JSON.');
    ancestors.add(item);
    try {
      if (Array.isArray(item)) {
        if (item.length > 4096 || Reflect.ownKeys(item).length !== item.length + 1) throw new Error('Array is not plain JSON.');
        const values: DatasetJson[] = [];
        for (let index = 0; index < item.length; index++) {
          const property = Object.getOwnPropertyDescriptor(item, String(index));
          if (!property || !('value' in property) || !property.enumerable) throw new Error('Array is not plain JSON.');
          values.push(visit(property.value, depth + 1));
        }
        return values;
      }
      const entries: Array<[string, DatasetJson]> = [];
      for (const key of Reflect.ownKeys(item)) {
        if (typeof key !== 'string') throw new Error('Symbol data cannot be captured.');
        if (response && depth === 0 && ['sdkHttpResponse', '_request_id'].includes(key)) continue;
        if (transportKeys.test(key) || ['__proto__', 'prototype', 'constructor'].includes(key)) throw new Error('Transport data cannot be captured.');
        const descriptor = Object.getOwnPropertyDescriptor(item, key);
        if (!descriptor || !('value' in descriptor) || !descriptor.enumerable) throw new Error('Hidden or computed data cannot be captured.');
        if (descriptor.value === undefined) continue;
        bytes += Buffer.byteLength(key);
        if (bytes > MAX_BYTES) throw new Error('Provider capture exceeds size limit.');
        entries.push([key, visit(descriptor.value, depth + 1)]);
      }
      return Object.fromEntries(entries);
    } finally {
      ancestors.delete(item);
    }
  };
  return visit(value, 0);
}

function safeBody(value: unknown, response = false): DatasetJson {
  const body = providerData(value, response);
  const encoded = JSON.stringify(body);
  if (Buffer.byteLength(encoded) > MAX_BYTES || mediaReferences(body, 'replay').length ||
      JSON.stringify(sanitizeMediaPayload(body, 'replay')) !== encoded) throw new Error('Provider data is not losslessly capturable.');
  return body;
}

function digest(operation: ReplayProviderOperation, input: DatasetJson, output: DatasetJson) {
  return createHash('sha256').update(JSON.stringify({ version: 1, operation, input, output })).digest('hex');
}

/** Snapshot the request before caller/provider mutation; absence is an honest cache miss. */
export function prepareProviderCapture(
  span: Span,
  operation: ReplayProviderOperation,
  request: unknown,
  supported: boolean,
): (response: unknown) => void {
  if (!supported || !span.isRecording()) return () => undefined;
  let input: DatasetJson;
  try {
    input = safeBody(request);
  } catch {
    return () => undefined;
  }
  return response => {
    try {
      const output = safeBody(response, true);
      if (!isCompleteProviderResponse(operation, output)) return;
      const encoded = JSON.stringify({ version: 1, operation, input, output, digest: digest(operation, input, output) });
      if (Buffer.byteLength(encoded) <= MAX_BYTES) span.setAttribute(MODEL_CAPTURE_ATTRIBUTE, encoded);
    } catch {
      // Telemetry must not alter the provider result or failure.
    }
  };
}

export function readProviderCapture(value: unknown) {
  if (typeof value !== 'string' || Buffer.byteLength(value) > MAX_BYTES) throw new Error('Missing complete provider capture.');
  const saved = recording.parse(JSON.parse(value));
  const input = safeBody(saved.input);
  const output = safeBody(saved.output);
  if (digest(saved.operation, input, output) !== saved.digest || !isCompleteProviderResponse(saved.operation, output)) throw new Error('Provider capture changed or is incomplete.');
  return { name: saved.operation, input, output };
}
