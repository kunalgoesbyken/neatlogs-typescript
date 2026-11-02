/**
 * Neatlogs OpenAI Agents SDK trace processor.
 *
 * Usage:
 *   import { openaiAgentsProcessor } from 'neatlogs/openai-agents';
 *   import { addTraceProcessor } from '@openai/agents';
 *   addTraceProcessor(openaiAgentsProcessor());
 *
 * Creates spans: WORKFLOW (traces), AGENT (agent runs), LLM (generations), TOOL (function calls).
 */

import { trace, SpanStatusCode, type Span } from '@opentelemetry/api';
import { getNeatlogsTracer, getNeatlogsBaseContext } from './core/provider.js';
import { captureMedia } from './core/media.js';
import { recordFailure } from './core/span-outcome.js';

const TRACER_NAME = 'neatlogs.openai_agents';

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export function openaiAgentsProcessor(): any {
  return new NeatlogsTraceProcessor();
}

// ---------------------------------------------------------------------------
// Trace Processor (implements OpenAI Agents SDK TracingProcessor protocol)
// ---------------------------------------------------------------------------

class NeatlogsTraceProcessor {
  private _spans: Map<string, Span> = new Map();
  private _startTimes: Map<string, number> = new Map();
  private _rootInputDone: Set<string> = new Set();
  // Fingerprints of input messages already captured per trace, so a long
  // conversation is not copied onto every response span. Only a verified
  // shared prefix is skipped: a same-length list whose contents differ is new
  // input, not an already-seen prefix (parallel branches, fixed-size or reset
  // histories), so the prefix is recomputed from the fingerprints each time.
  private _inputSeen: Map<string, string[]> = new Map();
  // First error per trace from a span whose failure ends the run.
  private _runErrors: Map<string, unknown> = new Map();
  // Tool/function spans and everything nested under them: the run carries on
  // after a tool fails, so their errors never decide the root status.
  private _toolScoped: Set<string> = new Set();

  // The @openai/agents SDK passes a Trace object: { traceId, name, groupId, metadata }.
  onTraceStart(traceData: any): void {
    const tracer = getNeatlogsTracer(TRACER_NAME);
    const attrs: Record<string, any> = { 'neatlogs.span.kind': 'WORKFLOW' };

    const workflowName = traceData?.name ?? traceData?.workflow_name;
    if (workflowName) attrs['neatlogs.workflow.name'] = workflowName;

    const traceId = traceData?.traceId ?? traceData?.trace_id;
    if (traceId) attrs['neatlogs.agent.trace_id'] = String(traceId);

    const span = tracer.startSpan('openai_agents.trace', { attributes: attrs }, getNeatlogsBaseContext());
    const key = String(traceId ?? `trace_${Date.now()}`);
    this._spans.set(key, span);
    this._startTimes.set(key, Date.now());
  }

  onTraceEnd(traceData: any): void {
    this._endTrace(String(traceData?.traceId ?? traceData?.trace_id ?? ''));
  }

  // Ends the trace's root span once. The SDK never calls onTraceEnd when
  // run() throws, so a failed run ends it from its closing `task` span; the
  // later onTraceEnd, if any, finds nothing left to end.
  private _endTrace(key: string): void {
    const span = this._spans.get(key);
    if (!span) return;

    const startTime = this._startTimes.get(key);
    if (startTime) {
      span.setAttribute('neatlogs.metrics.duration_ms', Date.now() - startTime);
    }

    // The SDK's Trace object carries no outcome, so the root reflects the
    // run-ending error recorded while its spans closed.
    if (this._runErrors.has(key)) {
      recordFailure(span, this._runErrors.get(key));
    } else {
      span.setStatus({ code: SpanStatusCode.OK });
    }
    span.end();
    this._spans.delete(key);
    this._startTimes.delete(key);
    this._rootInputDone.delete(key);
    this._inputSeen.delete(key);
    this._runErrors.delete(key);
  }

  // The SDK passes a Span object: { type, spanId, traceId, parentId?, spanData: {...} }.
  // The meaningful payload (type, name, input, output, usage, ...) lives in `spanData`.
  onSpanStart(span: any): void {
    const tracer = getNeatlogsTracer(TRACER_NAME);
    const data = span?.spanData ?? span ?? {};
    const spanType = data?.type ?? data?.span_type ?? span?.type ?? '';
    const spanId = String(span?.spanId ?? span?.span_id ?? data?.span_id ?? `span_${Date.now()}`);

    // Parent context: nest under the trace span (or a parent span) so the tree
    // forms. Base is ROOT_CONTEXT in isolated mode so a foreign provider's
    // active span can never leak in as our ancestor.
    const base = getNeatlogsBaseContext();
    const parentKey = String(span?.parentId ?? span?.traceId ?? span?.trace_id ?? '');
    const parentSpan = this._spans.get(parentKey);
    const parentCtx = parentSpan ? trace.setSpan(base, parentSpan) : base;

    let otelSpan: Span;

    if (spanType === 'agent' || spanType === 'agent_run') {
      const agentName = data?.name ?? data?.agent_name ?? 'agent';
      const attrs: Record<string, any> = {
        'neatlogs.span.kind': 'AGENT',
        'neatlogs.agent.name': agentName,
      };
      if (Array.isArray(data?.tools) && data.tools.length) {
        attrs['neatlogs.agent.available_tools'] = data.tools.join(',');
      }
      otelSpan = tracer.startSpan(`openai_agents.agent.${agentName}`, { attributes: attrs }, parentCtx);

    } else if (spanType === 'response' || spanType === 'generation' || spanType === 'llm') {
      const attrs: Record<string, any> = {
        'neatlogs.span.kind': 'LLM',
        'neatlogs.llm.provider': 'openai',
      };

      const model = data?.model;
      if (model) attrs['neatlogs.llm.model_name'] = model;

      const inputMsgs = data?.input ?? data?.messages;
      if (inputMsgs && Array.isArray(inputMsgs)) {
        for (let i = 0; i < inputMsgs.length; i++) {
          const msg = inputMsgs[i];
          const role = typeof msg === 'object' ? (msg.role ?? '') : '';
          const content = typeof msg === 'object' ? (msg.content ?? '') : String(msg);
          if (role) attrs[`neatlogs.llm.input_messages.${i}.role`] = role;
          if (content) attrs[`neatlogs.llm.input_messages.${i}.content`] = (typeof content === 'string' ? content : safeStringify(content));
        }
      }

      otelSpan = tracer.startSpan('openai_agents.generation', { attributes: attrs }, parentCtx);

    } else if (isToolSpanType(spanType)) {
      const toolName = data?.name ?? data?.function_name ?? 'tool';
      const attrs: Record<string, any> = {
        'neatlogs.span.kind': 'TOOL',
        'neatlogs.tool.name': toolName,
      };

      const toolInput = data?.input ?? data?.arguments;
      if (toolInput !== undefined) {
        attrs['input.value'] = (typeof toolInput === 'string' ? toolInput : safeStringify(toolInput));
      }

      otelSpan = tracer.startSpan(`openai_agents.tool.${toolName}`, { attributes: attrs }, parentCtx);

    } else if (spanType === 'handoff') {
      const attrs: Record<string, any> = { 'neatlogs.span.kind': 'AGENT' };
      if (data?.from_agent) attrs['neatlogs.agent.handoff_from'] = String(data.from_agent);
      if (data?.to_agent) attrs['neatlogs.agent.name'] = String(data.to_agent);

      otelSpan = tracer.startSpan('openai_agents.handoff', { attributes: attrs }, parentCtx);

    } else {
      const attrs: Record<string, any> = { 'neatlogs.span.kind': 'CHAIN' };
      otelSpan = tracer.startSpan(`openai_agents.${spanType || 'span'}`, { attributes: attrs }, parentCtx);
    }

    if (isToolSpanType(spanType) || this._toolScoped.has(String(span?.parentId ?? ''))) {
      this._toolScoped.add(spanId);
    }

    this._spans.set(spanId, otelSpan);
    this._startTimes.set(spanId, Date.now());
  }

  onSpanEnd(span: any): void {
    const data = span?.spanData ?? span ?? {};
    const spanId = String(span?.spanId ?? span?.span_id ?? data?.span_id ?? '');
    const otelSpan = this._spans.get(spanId);
    if (!otelSpan) return;

    const spanType = data?.type ?? data?.span_type ?? span?.type ?? '';
    const startTime = this._startTimes.get(spanId);

    if (spanType === 'response' || spanType === 'generation' || spanType === 'llm') {
      // @openai/agents 'response' span nests the full Responses API object under
      // `_response` and the request under `_input`. Older shapes use output/usage/model directly.
      const resp = data?._response ?? data?.response ?? {};
      const model = data?.model ?? resp?.model;
      if (model) otelSpan.setAttribute('neatlogs.llm.model_name', model);

      // Output text: Responses API output[] has message items with content[].text
      const outputItems = data?.output ?? resp?.output;
      if (Array.isArray(outputItems)) {
        const text = safeOutputText(otelSpan, outputItems);
        if (text) {
          otelSpan.setAttribute('neatlogs.llm.output_messages.0.role', 'assistant');
          otelSpan.setAttribute('neatlogs.llm.output_messages.0.content', text);
        }
        // Tool-call items advertised in the response output
        const toolCalls = outputItems.filter((o: any) => o?.type === 'function_call');
        for (let i = 0; i < toolCalls.length; i++) {
          const tc = toolCalls[i];
          otelSpan.setAttribute(`neatlogs.llm.tool_calls.${i}.name`, tc.name ?? '');
          otelSpan.setAttribute(`neatlogs.llm.tool_calls.${i}.arguments`, typeof tc.arguments === 'string' ? tc.arguments : safeStringify(tc.arguments ?? {}));
          if (tc.callId ?? tc.id) otelSpan.setAttribute(`neatlogs.llm.tool_calls.${i}.id`, tc.callId ?? tc.id);
        }
      } else if (outputItems?.content) {
        otelSpan.setAttribute('neatlogs.llm.output_messages.0.role', 'assistant');
        const c = outputItems.content;
        otelSpan.setAttribute('neatlogs.llm.output_messages.0.content', (typeof c === 'string' ? c : safeStringify(c)));
      }

      // The SDK fills response input at span end (`_input` in current releases).
      // Capture it on the LLM span and mirror the first user turn plus latest
      // assistant output onto the WORKFLOW root, which drives trace-list I/O.
      const inputItems = data?._input ?? data?.input;
      const traceKey = String(span?.traceId ?? span?.trace_id ?? '');
      const messages = Array.isArray(inputItems) ? inputItems : (typeof inputItems === 'string' ? [inputItems] : []);
      const fingerprints = messages.map(messageFingerprint);
      // Only the turns past the verified shared prefix with what earlier
      // response spans of this trace already captured.
      const previous = this._inputSeen.get(traceKey) ?? [];
      let start = 0;
      while (start < previous.length && start < fingerprints.length && previous[start] === fingerprints[start]) start++;
      const inputTurns = captureInputMessages(otelSpan, inputItems, start);
      this._inputSeen.set(traceKey, fingerprints);
      const root = this._spans.get(traceKey);
      if (root) {
        if (!this._rootInputDone.has(traceKey)) {
          const userTurn = latestUserTurn(inputTurns);
          if (userTurn) {
            root.setAttribute('input.value', userTurn.slice(0, 10000));
            this._rootInputDone.add(traceKey);
          }
        }
        const outputText = Array.isArray(outputItems) ? safeOutputText(otelSpan, outputItems) : assistantOutputText(outputItems);
        if (outputText) root.setAttribute('output.value', outputText.slice(0, 10000));
      }

      const usage = data?.usage ?? resp?.usage;
      if (usage) {
        const inputTokens = usage.input_tokens ?? usage.prompt_tokens ?? usage.inputTokens;
        const outputTokens = usage.output_tokens ?? usage.completion_tokens ?? usage.outputTokens;
        const totalTokens = usage.total_tokens ?? usage.totalTokens;
        if (inputTokens != null) otelSpan.setAttribute('neatlogs.llm.token_count.prompt', inputTokens);
        if (outputTokens != null) otelSpan.setAttribute('neatlogs.llm.token_count.completion', outputTokens);
        if (totalTokens != null) otelSpan.setAttribute('neatlogs.llm.token_count.total', totalTokens);
        else if (inputTokens != null && outputTokens != null) {
          otelSpan.setAttribute('neatlogs.llm.token_count.total', inputTokens + outputTokens);
        }
      }

    } else if (spanType === 'function' || spanType === 'tool' || spanType === 'tool_call') {
      const output = data?.output ?? data?.result;
      if (output != null) {
        otelSpan.setAttribute('output.value', (typeof output === 'string' ? output : safeStringify(output)));
      }

    } else if (spanType === 'agent' || spanType === 'agent_run') {
      const output = data?.output;
      if (output != null) {
        otelSpan.setAttribute('output.value', (typeof output === 'string' ? output : safeStringify(output)));
      }
    }

    const error = reportedError(data?.error ?? span?.error);
    if (error) {
      // The SDK reports `{ message, data }`, not an Error.
      recordFailure(otelSpan, error);
      const traceKey = String(span?.traceId ?? span?.trace_id ?? '');
      // A handoff error leaves the current agent running, like a tool error.
      if (
        traceKey &&
        spanType !== 'handoff' &&
        !this._toolScoped.has(spanId) &&
        !this._runErrors.has(traceKey)
      ) {
        this._runErrors.set(traceKey, error);
      }
    } else {
      otelSpan.setStatus({ code: SpanStatusCode.OK });
    }

    if (startTime) {
      otelSpan.setAttribute('neatlogs.llm.metrics.duration_ms', Date.now() - startTime);
    }

    otelSpan.end();
    this._spans.delete(spanId);
    this._startTimes.delete(spanId);
    this._toolScoped.delete(spanId);

    if (error && spanType === 'task') {
      this._endTrace(String(span?.traceId ?? span?.trace_id ?? ''));
    }
  }

  shutdown(): void {
    for (const [, span] of this._spans) {
      span.setStatus({ code: SpanStatusCode.ERROR, message: 'Processor shutdown before span completed' });
      span.end();
    }
    this._spans.clear();
    this._startTimes.clear();
    this._rootInputDone.clear();
    this._inputSeen.clear();
    this._runErrors.clear();
    this._toolScoped.clear();
  }

  forceFlush(): void {}
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

// Captures messages from index `start` on. Earlier ones were already written to
// the previous response span of this trace. Indexes stay the same as in the
// request, so `input_messages.N` always means message N. Every message is
// written in full, like the other adapters; a span too large for one export
// request is handled by the exporter, not cut here.
function captureInputMessages(span: Span, input: any, start = 0): Array<{ role: string; text: string }> {
  const messages = Array.isArray(input) ? input : (typeof input === 'string' ? [{ role: 'user', content: input }] : []);
  const turns: Array<{ role: string; text: string }> = [];
  for (let i = 0; i < messages.length; i++) {
    if (i < start) {
      turns.push({ role: '', text: '' });
      continue;
    }
    const message = messages[i];
    const role = typeof message === 'object' && message ? String(message.role ?? '') : '';
    const content = typeof message === 'object' && message ? message.content : message;
    // Route media through typed capture first, then read text parts only, so
    // inline bytes never reach text attributes.
    const safeContent = captureMedia(span, `neatlogs.llm.input_messages.${i}`, content, 'input');
    const text = contentText(safeContent);
    turns.push({ role, text });
    if (role) span.setAttribute(`neatlogs.llm.input_messages.${i}.role`, role);
    if (text) span.setAttribute(`neatlogs.llm.input_messages.${i}.content`, text);
  }
  return turns;
}

// Last user turn, taken from the already sanitized text.
function latestUserTurn(turns: Array<{ role: string; text: string }>): string {
  for (let i = turns.length - 1; i >= 0; i--) {
    if (turns[i].role === 'user') return turns[i].text;
  }
  return '';
}

// Assistant text from a Responses output[] list, with media routed through capture first.
function safeOutputText(span: Span, output: any[]): string {
  const parts = output
    .filter((item: any) => item?.type === 'message' || item?.role === 'assistant')
    .flatMap((item: any) => (Array.isArray(item.content) ? item.content : [item.content]));
  const safe = captureMedia(span, 'neatlogs.llm.output_messages.0', parts, 'output');
  return (Array.isArray(safe) ? safe : [])
    .map((c: any) => (typeof c === 'string' ? c : c?.text ?? ''))
    .join('');
}

function assistantOutputText(output: any): string {
  if (!Array.isArray(output)) return output?.content ? contentText(output.content) : '';
  return output
    .filter((item: any) => item?.type === 'message' || item?.role === 'assistant')
    .flatMap((item: any) => Array.isArray(item.content) ? item.content : [item.content])
    .map((content: any) => typeof content === 'string' ? content : content?.text ?? '')
    .join('');
}

// 53-bit string hash (cyrb53) for input-prefix fingerprints.
function cyrb53(str: string, seed = 0): string {
  let h1 = 0xdeadbeef ^ seed, h2 = 0x41c6ce57 ^ seed;
  for (let i = 0, ch; i < str.length; i++) {
    ch = str.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return (h2 >>> 0).toString(36) + (h1 >>> 0).toString(36);
}

// Identity of one input message for prefix verification: role, text, and for
// every non-text part a hash of its type and reference or payload, so two
// messages with the same text but different media are not the same message.
// Only the hash is kept; inline bytes never reach an attribute.
function messageFingerprint(message: any): string {
  const role = typeof message === 'object' && message ? String(message.role ?? '') : '';
  const content = typeof message === 'object' && message ? message.content : message;
  const identity = Array.isArray(content)
    ? content.map(partIdentity)
    : (typeof content === 'string' ? content : partIdentity(content));
  return cyrb53(safeStringify([role, identity]));
}

function partIdentity(part: any): unknown {
  if (typeof part === 'string') return ['text', part];
  if (!part || typeof part !== 'object') return ['empty'];
  const { text, ...rest } = part;
  const media = Object.keys(rest).length > 1 || (Object.keys(rest).length === 1 && !('type' in rest))
    ? cyrb53(safeStringify(rest))
    : '';
  return [part.type ?? '', typeof text === 'string' ? text : '', media];
}

// Text only: media parts are captured by captureMedia, never stringified here.
function contentText(content: any): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .map((part: any) => (typeof part === 'string' ? part : typeof part?.text === 'string' ? part.text : ''))
      .join('');
  }
  return typeof content?.text === 'string' ? content.text : '';
}

// The SDK reports a failure as `{ message: "Error in agent run", data: { error: "Error: model down" } }`;
// the cause is in `data.error`, so use it when it is there.
function reportedError(error: any): any {
  const cause = error?.data?.error;
  return typeof cause === 'string' && cause ? { ...error, message: cause } : error;
}

function isToolSpanType(spanType: string): boolean {
  return spanType === 'function' || spanType === 'tool' || spanType === 'tool_call';
}

function safeStringify(value: unknown): string {
  try {
    return typeof value === 'string' ? value : JSON.stringify(value);
  } catch {
    return '';
  }
}
