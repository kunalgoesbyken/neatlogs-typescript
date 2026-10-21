/**
 * PII masking support for Neatlogs spans.
 *
 * Users supply a callable that receives the full span dict and returns
 * the (possibly modified) span dict. Return null to drop the span entirely.
 */

import type { MaskFunction } from '../types.js';
import { getLogger } from './logger.js';

const logger = getLogger();
export const DEFAULT_MASK_TIMEOUT_MS = 5_000;

export interface ApplyMaskOptions {
  signalType?: 'span' | 'log';
  timeoutMs?: number;
}

/** Module-level registry: key -> mask function */
const _MASK_REGISTRY = new Map<string, MaskFunction>();
const _MASK_RESULTS = new WeakMap<object, Promise<Record<string, any> | null>>();

let _nextId = 0;

/**
 * Register a mask function and return its lookup key.
 */
export function registerMask(fn: MaskFunction): string {
  const key = String(++_nextId);
  _MASK_REGISTRY.set(key, fn);
  return key;
}

/**
 * Apply the effective mask function to span data.
 *
 * Per-span mask (stored in attributes["neatlogs.mask_id"]) takes
 * precedence over the global mask. Returns the (possibly modified) dict.
 */
function effectiveMask(
  spanData: Record<string, any>,
  globalMask: MaskFunction | null | undefined,
): MaskFunction | null {
  const maskId = spanData?.attributes?.['neatlogs.mask_id'];
  let maskFn: MaskFunction | undefined | null = null;

  if (maskId) {
    maskFn = _MASK_REGISTRY.get(String(maskId)) ?? null;
  }

  if (!maskFn) {
    maskFn = globalMask ?? null;
  }

  return maskFn;
}

export async function applyMask(
  spanData: Record<string, any>,
  globalMask: MaskFunction | null | undefined,
  options: ApplyMaskOptions = {},
): Promise<Record<string, any> | null> {
  const maskFn = effectiveMask(spanData, globalMask);
  if (!maskFn) return spanData;
  // Existing callbacks may mask only normalized I/O; never retain a second raw copy.
  if (spanData.attributes) delete spanData.attributes['neatlogs.replay.model_capture'];

  const signalType = options.signalType ?? 'span';
  const timeoutMs = options.timeoutMs ?? DEFAULT_MASK_TIMEOUT_MS;
  const controller = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;

  try {
    // Run user masking outside SpanProcessor.onEnd() so a slow callback does
    // not block application request completion.
    const maskPromise = Promise.resolve().then(() =>
      maskFn(spanData, { signal: controller.signal, signalType, timeoutMs }),
    );
    const timeoutPromise = new Promise<never>((_resolve, reject) => {
      timer = setTimeout(() => {
        controller.abort();
        reject(new Error(`mask callback exceeded ${timeoutMs}ms`));
      }, timeoutMs);
      timer.unref?.();
    });
    const result = await Promise.race([maskPromise, timeoutPromise]);
    // null means "drop this span entirely"; undefined means "keep original"
    if (result === null) return null;
    return result !== undefined ? result : spanData;
  } catch (exc) {
    const signalName = spanData?.name ?? spanData?.signal_type ?? signalType;
    logger.error(
      `mask callable failed for ${signalType} '${signalName}': ${exc} — dropping it to prevent unmasked export.`,
    );
    return null;
  } finally {
    if (timer !== undefined) clearTimeout(timer);
    controller.abort();
  }
}

/** Schedule and retain one masking result for the exporter that receives span. */
export function scheduleMask(
  span: object,
  spanData: Record<string, any>,
  globalMask: MaskFunction | null | undefined,
): Promise<Record<string, any> | null> {
  const result = applyMask(spanData, globalMask, { signalType: 'span' });
  _MASK_RESULTS.set(span, result);
  return result;
}

/** Return the result scheduled by NeatlogsSpanProcessor for this snapshot. */
export function getScheduledMask(span: object): Promise<Record<string, any> | null> | undefined {
  return _MASK_RESULTS.get(span);
}

/**
 * Clear all registered masks. Used for testing.
 * @internal
 */
export function _clearMaskRegistry(): void {
  _MASK_REGISTRY.clear();
  _nextId = 0;
}
