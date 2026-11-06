import type { Span } from '@opentelemetry/api';
import { z } from 'zod';
import type { DatasetJson } from '../dataset/api.js';
import { hasReplaySession, lookupReplay, reserveReplay } from './session.js';
import { aliasMediaCaptureSpan } from '../core/media.js';

const chatResponse = z.object({
  id: z.string(), object: z.literal('chat.completion'), created: z.number().finite(), model: z.string(),
  choices: z.array(z.object({ index: z.number().int(), finish_reason: z.string().nullable(), message: z.object({ role: z.literal('assistant') }).passthrough() })),
}).passthrough();
const responsesResponse = z.object({
  id: z.string(), object: z.literal('response'), created_at: z.number().finite(), model: z.string(),
  status: z.literal('completed'), output: z.array(z.record(z.string(), z.unknown())), output_text: z.string(),
}).passthrough();
const googleResponse = z.object({
  candidates: z.array(z.object({ content: z.object({ parts: z.array(z.record(z.string(), z.unknown())) }).passthrough() }).passthrough()).min(1),
}).passthrough();

export const replayProviderOperationSchema = z.enum(['openai.chat.completions.create', 'openai.responses.create', 'google_genai.models.generate_content']);
export type ReplayProviderOperation = z.infer<typeof replayProviderOperationSchema>;
const responses = {
  'openai.chat.completions.create': chatResponse,
  'openai.responses.create': responsesResponse,
  'google_genai.models.generate_content': googleResponse,
};

export function isCompleteProviderResponse(operation: ReplayProviderOperation, value: unknown): boolean {
  return responses[operation].safeParse(value).success;
}

const MAX_PROVIDER_BYTES = 1024 * 1024;
const transportKeys = /^(?:api_?key|authorization|proxyAuthorization|headers|extraHeaders|httpOptions|abortSignal|credentials|access_token|client_secret)$/i;

/** Copy actual data properties, never SDK getters, toJSON hooks, or reconstructed messages. */
export function providerData(value: unknown, response = false): DatasetJson {
  let bytes = 0;
  let nodes = 0;
  const ancestors = new Set<object>();
  const visit = (item: unknown, depth: number): DatasetJson => {
    if (++nodes > 4096 || depth > 32) throw new Error('Provider capture exceeds traversal limit.');
    if (typeof item === 'string') {
      bytes += Buffer.byteLength(item);
      if (bytes > MAX_PROVIDER_BYTES) throw new Error('Provider capture exceeds size limit.');
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
        if (bytes > MAX_PROVIDER_BYTES) throw new Error('Provider capture exceeds size limit.');
        entries.push([key, visit(descriptor.value, depth + 1)]);
      }
      return Object.fromEntries(entries);
    } finally {
      ancestors.delete(item);
    }
  };
  return visit(value, 0);
}

/** Match provider serialization while rejecting invalid values and array elements. */
export function providerReplayInput(value: unknown): DatasetJson {
  return providerData(value);
}

/** Keep API-call order independent of when a caller consumes its promise. */
export function prepareProviderReplay(operation: ReplayProviderOperation, input: unknown, supported: boolean): { canRestore: boolean; consume: (restore: boolean) => unknown | undefined } | null {
  if (!hasReplaySession()) return null;
  let request: DatasetJson | undefined;
  if (supported) { try { request = providerReplayInput(input); } catch { /* Unsupported requests stay live. */ } }
  const reservation = reserveReplay('model', operation, request, value => isCompleteProviderResponse(operation, value))!;
  return { canRestore: reservation.canRestore, consume: restore => { const decision = reservation.consume(restore); return decision.outcome === 'HIT' ? decision.entry.output : undefined; } };
}

/** Missing, incomplete and unsupported recordings make the remaining run live. */
export function providerReplay(operation: ReplayProviderOperation, input: unknown, supported: boolean): unknown | undefined {
  if (!hasReplaySession()) return undefined;
  let request: DatasetJson | undefined;
  if (supported) {
    try { request = providerReplayInput(input); } catch { /* Non-JSON request must execute normally. */ }
  }
  const decision = lookupReplay('model', operation, request, value => isCompleteProviderResponse(operation, value));
  return decision?.outcome === 'HIT' ? decision.entry.output : undefined;
}

/** Retain the saved response for application code, but count no new billed tokens. */
export function cachedModelSpan(span: Span): Span {
  for (const metric of ['prompt', 'completion', 'total', 'reasoning', 'cache_read', 'cache_write'])
    span.setAttribute(`neatlogs.llm.token_count.${metric}`, 0);
  const facade = new Proxy(span, {
    get(target, property) {
      if (property === 'setAttribute') return (key: string, value: Parameters<Span['setAttribute']>[1]) =>
        target.setAttribute(key, /^neatlogs\.llm\.(token_count\.|cost)/.test(key) ? 0 : value);
      const value = Reflect.get(target, property, target);
      return typeof value === 'function' ? value.bind(target) : value;
    },
  });
  aliasMediaCaptureSpan(facade, span);
  return facade;
}

export async function restoreGoogleResponse(saved: unknown): Promise<unknown> {
  const parsed = googleResponse.parse(saved);
  // Use the installed SDK's getters for text, functionCalls and structured results.
  const { GenerateContentResponse } = await import('@google/genai');
  const response = new GenerateContentResponse();
  for (const [key, value] of Object.entries(parsed)) {
    if (['__proto__', 'prototype', 'constructor'].includes(key)) throw new Error('Invalid saved model response.');
    Object.defineProperty(response, key, { value, enumerable: true, writable: true, configurable: true });
  }
  return response;
}
