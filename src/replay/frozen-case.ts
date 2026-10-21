import { z } from 'zod';
import { MODEL_CAPTURE_ATTRIBUTE, readProviderCapture } from './provider-capture.js';
import { parseReplayRecording, type ReplayEntry, type ReplayRecording } from './contracts.js';

const object = z.record(z.string(), z.unknown());
const missing = new Set(['unavailable', 'retentionExpired', 'truncated', 'previewOnly', 'redacted', 'masked']);
function record(value: unknown): Record<string, unknown> {
  return object.parse(typeof value === 'string' ? JSON.parse(value) : value ?? {});
}
function storedJson(value: unknown): z.infer<ReturnType<typeof z.json>> {
  if (typeof value === 'string') {
    try { return z.json().parse(JSON.parse(value)); } catch { return value; }
  }
  return z.json().parse(value);
}
function complete(span: Record<string, unknown>, attributes: Record<string, unknown>): boolean {
  const status = typeof span.status === 'object' && span.status !== null ? record(span.status).code : span.status;
  return ![span.payload_state, span.input_content_state, span.output_content_state].some(value => typeof value === 'string' && missing.has(value)) &&
    ![span.redacted, span.masked, attributes['pii.server_redacted'], attributes['pii.server_redaction_failed'], attributes['neatlogs.payload.truncated'], attributes['neatlogs.capture.truncated'], attributes['neatlogs.stream.incomplete']].some(value => value === true || value === 'true') &&
    !missing.has(String(attributes['neatlogs.capture_fidelity'])) &&
    attributes['pii.server_redaction_status'] !== 'redacted' &&
    ![status, span.status_code].some(value => value === 'ERROR' || value === 2 || value === '2') &&
    (span.error == null || span.error === '') && (span.error_message == null || span.error_message === '');
}

/** Only verified raw provider captures can restore model SDK responses; normalized text cannot. */
export function replayRecordingFromCase(value: unknown): { recording: ReplayRecording; skippedTools: number } {
  const item = object.parse(value);
  const source = z.object({ traceId: z.string().min(1) }).parse(item.source);
  const context = z.object({ spans: z.array(object).max(5000) }).parse(item.context);
  const entries: ReplayEntry[] = [];
  const identities = new Set<string>();
  let skippedTools = 0;
  for (const span of context.spans) {
    let semantic: Record<string, unknown>;
    try { semantic = record(span.semantic); } catch { skippedTools++; continue; }
    const kind = semantic.kind ?? span.span_type;
    if (kind === 'LLM') {
      try {
        const attributes = record(span.attributes);
        if (!complete(span, attributes)) continue;
        const spanId = z.string().min(1).max(200).parse(span.span_id);
        entries.push({ spanId, kind: 'model', ...readProviderCapture(attributes[MODEL_CAPTURE_ATTRIBUTE]) });
      } catch { /* Unsupported model evidence stays live. */ }
      continue;
    }
    if (kind !== 'TOOL' && kind !== 'MCP_TOOL') continue;
    try {
      const attributes = record(span.attributes);
      if (!complete(span, attributes)) throw new Error('Incomplete recording.');
      const spanId = z.string().min(1).max(200).parse(span.span_id);
      if (identities.has(spanId)) throw new Error('Duplicate tool span.');
      identities.add(spanId);
      let name: unknown, input: unknown, output: unknown;
      if (semantic.kind) {
        const call = record(semantic.call), result = record(semantic.result), definition = record(semantic.definition);
        if (!call.id || result.call_id !== call.id || definition.name !== call.name || result.is_error !== false ||
            !Object.hasOwn(result, 'value') || Array.isArray(result.media) && result.media.length)
          throw new Error('Incomplete tool result.');
        name = call.name; input = z.json().parse(call.arguments); output = z.json().parse(result.value);
      } else {
        name = span.tool_name ?? attributes['neatlogs.tool.name'] ?? attributes['neatlogs.raw.tool.name'];
        input = storedJson(attributes['neatlogs.tool.input'] ?? attributes['neatlogs.raw.tool.input'] ?? span.tool_input ?? span.input_value);
        output = storedJson(attributes['neatlogs.tool.output'] ?? attributes['neatlogs.raw.tool.output'] ?? span.tool_output ?? span.output_value);
      }
      entries.push({ spanId, kind: 'tool', name: z.string().min(1).max(200).parse(name), input: z.json().parse(input), output: z.json().parse(output) });
    } catch { skippedTools++; }
  }
  // Ambiguous duplicate identities invalidate all copies, not just the last one.
  const counts = new Map<string, number>();
  for (const span of context.spans) if (typeof span.span_id === 'string') counts.set(span.span_id, (counts.get(span.span_id) ?? 0) + 1);
  const valid = entries.filter(entry => counts.get(entry.spanId) === 1);
  skippedTools += entries.filter(entry => entry.kind === 'tool' && counts.get(entry.spanId) !== 1).length;
  return { recording: parseReplayRecording({ version: 1, sourceTraceId: source.traceId, entries: valid }), skippedTools };
}
