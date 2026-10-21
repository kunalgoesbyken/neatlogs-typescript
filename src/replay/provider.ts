import type { Span } from '@opentelemetry/api';
import { z } from 'zod';
import { json, type DatasetJson } from '../dataset/api.js';
import { hasReplaySession, lookupReplay } from './session.js';
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

/** A provider request body, without transport options or a lossy JSON conversion. */
export function providerReplayInput(value: unknown): DatasetJson {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return json(value);
  if (Object.getPrototypeOf(value) !== Object.prototype && Object.getPrototypeOf(value) !== null) return json(value);
  // Provider SDKs omit undefined optional object fields when serializing a body.
  const defined = Object.fromEntries(Object.entries(value).filter(([, item]) => item !== undefined));
  return json(defined);
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
