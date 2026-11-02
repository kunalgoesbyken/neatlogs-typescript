import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest';
import { createHash } from 'node:crypto';
import {
  trace as otelTrace,
  context as otelContext,
} from '@opentelemetry/api';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import {
  SimpleSpanProcessor,
  InMemorySpanExporter,
  type ReadableSpan,
} from '@opentelemetry/sdk-trace-base';
import type { ChatCompletionContentPart } from 'openai/resources/chat/completions';
import type { ResponseInputFile } from 'openai/resources/responses/responses';

import { wrapOpenAI, traceTool as traceToolOpenAI } from '../../src/openai.js';
import { wrapAnthropic, traceTool as traceToolAnthropic } from '../../src/anthropic.js';
import { langchainHandler } from '../../src/langchain.js';
import { strandsHooks } from '../../src/strands.js';
import { openaiAgentsProcessor } from '../../src/openai-agents.js';
import { wrapMastra } from '../../src/mastra-wrap.js';
import { _setNeatlogsProvider } from '../../src/core/provider.js';
import { discardPendingMedia } from '../../src/core/media.js';

let provider: NodeTracerProvider;
let exporter: InMemorySpanExporter;

let prevAutoRoot: string | undefined;

beforeAll(() => {
  // These assert LLM/TOOL attribute mapping on bare wrappers. Auto-root would
  // add a WORKFLOW parent span; disable it so span counts reflect the wrapper.
  prevAutoRoot = process.env.NEATLOGS_AUTO_ROOT;
  process.env.NEATLOGS_AUTO_ROOT = 'false';
  exporter = new InMemorySpanExporter();
  provider = new NodeTracerProvider();
  provider.addSpanProcessor(new SimpleSpanProcessor(exporter));
  _setNeatlogsProvider(provider);
});

afterAll(async () => {
  _setNeatlogsProvider(null);
  await provider.shutdown();
  if (prevAutoRoot === undefined) delete process.env.NEATLOGS_AUTO_ROOT;
  else process.env.NEATLOGS_AUTO_ROOT = prevAutoRoot;
});

beforeEach(() => {
  exporter.reset();
});

function getSpans(): ReadableSpan[] {
  return exporter.getFinishedSpans();
}

function attr(span: ReadableSpan, key: string): any {
  return span.attributes[key];
}

// ---------------------------------------------------------------------------
// wrapOpenAI
// ---------------------------------------------------------------------------

describe('wrapOpenAI', () => {
  it('traces non-streaming chat.completions.create', async () => {
    const fakeResponse = {
      id: 'chatcmpl-123',
      model: 'gpt-4o',
      choices: [{ message: { role: 'assistant', content: 'Hello!' }, finish_reason: 'stop' }],
      usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
    };

    const fakeClient = {
      chat: {
        completions: {
          create: async () => fakeResponse,
        },
      },
    };

    const wrapped = wrapOpenAI(fakeClient as any);
    const result = await wrapped.chat.completions.create({
      model: 'gpt-4o',
      messages: [{ role: 'user', content: 'Hi' }],
    });

    expect(result).toEqual(fakeResponse);

    const spans = getSpans();
    expect(spans.length).toBe(1);
    expect(attr(spans[0], 'neatlogs.span.kind')).toBe('LLM');
    expect(attr(spans[0], 'neatlogs.llm.model_name')).toBe('gpt-4o');
    expect(attr(spans[0], 'neatlogs.llm.input_messages.0.role')).toBe('user');
    expect(attr(spans[0], 'neatlogs.llm.input_messages.0.content')).toBe('Hi');
    expect(attr(spans[0], 'neatlogs.llm.output_messages.0.content')).toBe('Hello!');
    expect(attr(spans[0], 'neatlogs.llm.token_count.prompt')).toBe(10);
    expect(attr(spans[0], 'neatlogs.llm.token_count.completion')).toBe(5);
  });

  it('captures typed OpenAI file inputs from Chat Completions and Responses', async () => {
    const fileBytes = Buffer.from('report contents');
    const chatFiles = [
      {
        type: 'file',
        file: { file_id: 'file-chat-123' },
      },
      {
        type: 'file',
        file: {
          file_data: fileBytes.toString('base64'),
          filename: 'report.txt',
        },
      },
    ] satisfies ChatCompletionContentPart[];
    const responseFile = {
      type: 'input_file',
      file_url:
        'https://alice:password@example.com/report.pdf?X-Amz-Credential=secret&X-Amz-Signature=signature#fragment',
    } satisfies ResponseInputFile;
    const wrapped = wrapOpenAI({
      chat: {
        completions: {
          create: async () => ({ choices: [{ message: { content: 'ok' } }] }),
        },
      },
      responses: {
        create: async () => ({ output_text: 'ok', output: [] }),
      },
    } as any);

    await wrapped.chat.completions.create({
      model: 'gpt-4o',
      messages: [{ role: 'user', content: chatFiles }],
    });
    await wrapped.responses.create({
      model: 'gpt-5.1',
      input: [{ role: 'user', content: [responseFile] }],
    });

    const [chatSpan, responsesSpan] = getSpans();
    expect(attr(chatSpan, 'neatlogs.llm.input_messages.0.media.0.reference')).toBe(
      'file-chat-123',
    );
    expect(attr(chatSpan, 'neatlogs.llm.input_messages.0.media.0.type')).toBe('document');
    expect(attr(chatSpan, 'neatlogs.llm.input_messages.0.media.1.sha256')).toBe(
      createHash('sha256').update(fileBytes).digest('hex'),
    );
    expect(attr(responsesSpan, 'neatlogs.llm.input_messages.0.media.0.reference')).toBe(
      'https://example.com/report.pdf',
    );
    expect(
      JSON.stringify(
        Object.fromEntries(
          Object.entries(responsesSpan.attributes).filter(([key]) => key.includes('.media.')),
        ),
      ),
    ).not.toMatch(/alice|password|secret|signature|fragment/);
    expect(attr(responsesSpan, 'neatlogs.llm.input_messages.0.media.0.source')).toBe('url');
  });

  it('traces streaming chat.completions.create', async () => {
    const chunks = [
      { choices: [{ delta: { content: 'He' }, finish_reason: null }], model: 'gpt-4o' },
      { choices: [{ delta: { content: 'llo' }, finish_reason: null }], model: 'gpt-4o' },
      { choices: [{ delta: {}, finish_reason: 'stop' }], model: 'gpt-4o' },
      { choices: [], usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 } },
    ];

    async function* fakeStream() {
      for (const chunk of chunks) yield chunk;
    }

    const fakeClient = {
      chat: {
        completions: {
          create: async (opts: any) => {
            const iter = fakeStream();
            return { [Symbol.asyncIterator]: () => iter };
          },
        },
      },
    };

    const wrapped = wrapOpenAI(fakeClient as any);
    const stream = await wrapped.chat.completions.create({
      model: 'gpt-4o',
      messages: [{ role: 'user', content: 'Hi' }],
      stream: true,
    });

    const collected: any[] = [];
    for await (const chunk of stream) {
      collected.push(chunk);
    }

    expect(collected.length).toBe(4);

    const spans = getSpans();
    expect(spans.length).toBe(1);
    expect(attr(spans[0], 'neatlogs.llm.output_messages.0.content')).toBe('Hello');
    expect(attr(spans[0], 'neatlogs.llm.token_count.prompt')).toBe(5);
  });

  it('traces non-streaming responses.create output', async () => {
    const fakeResponse = {
      id: 'resp-123',
      model: 'gpt-5.1',
      status: 'completed',
      output_text: 'A complete response',
      output: [],
      usage: { input_tokens: 8, output_tokens: 3, total_tokens: 11 },
    };
    const wrapped = wrapOpenAI({
      responses: { create: async () => fakeResponse },
    } as any);

    const result = await wrapped.responses.create({ model: 'gpt-5.1', input: 'Hello' });

    expect(result).toEqual(fakeResponse);
    const spans = getSpans();
    expect(spans.length).toBe(1);
    expect(attr(spans[0], 'input.value')).toBe('Hello');
    expect(attr(spans[0], 'output.value')).toBe('A complete response');
    expect(attr(spans[0], 'neatlogs.llm.output_messages.0.content')).toBe('A complete response');
    expect(attr(spans[0], 'neatlogs.llm.token_count.total')).toBe(11);
  });

  it('keeps a streaming responses.create span open and captures its final output', async () => {
    const events = [
      { type: 'response.output_text.delta', delta: 'Hello ' },
      { type: 'response.output_text.delta', delta: 'world' },
      {
        type: 'response.completed',
        response: {
          model: 'gpt-5.1',
          status: 'completed',
          output_text: 'Hello world',
          output: [],
          usage: {
            input_tokens: 7,
            output_tokens: 2,
            total_tokens: 9,
            input_tokens_details: { cached_tokens: 4 },
            output_tokens_details: { reasoning_tokens: 1 },
          },
        },
      },
    ];
    async function* fakeStream() {
      for (const event of events) yield event;
    }
    const wrapped = wrapOpenAI({
      responses: {
        create: async () => {
          const iterator = fakeStream();
          return { [Symbol.asyncIterator]: () => iterator };
        },
      },
    } as any);

    const stream = await wrapped.responses.create({ model: 'gpt-5.1', input: 'Hi', stream: true });
    expect(getSpans()).toHaveLength(0);

    const collected: any[] = [];
    for await (const event of stream) collected.push(event);

    expect(collected).toEqual(events);
    const spans = getSpans();
    expect(spans).toHaveLength(1);
    expect(attr(spans[0], 'neatlogs.llm.is_streaming')).toBe(true);
    expect(attr(spans[0], 'output.value')).toBe('Hello world');
    expect(attr(spans[0], 'neatlogs.llm.output_messages.0.content')).toBe('Hello world');
    expect(attr(spans[0], 'neatlogs.llm.token_count.prompt')).toBe(7);
    expect(attr(spans[0], 'neatlogs.llm.token_count.completion')).toBe(2);
    expect(attr(spans[0], 'neatlogs.llm.token_count.total')).toBe(9);
    expect(attr(spans[0], 'neatlogs.llm.token_count.cache_read')).toBe(4);
    expect(attr(spans[0], 'neatlogs.llm.token_count.reasoning')).toBe(1);
  });

  it('traceTool wraps a function with TOOL span', async () => {
    const getWeather = traceToolOpenAI('get_weather', async (args: { city: string }) => {
      return `Sunny in ${args.city}`;
    });

    const result = await getWeather({ city: 'NYC' });
    expect(result).toBe('Sunny in NYC');

    const spans = getSpans();
    expect(spans.length).toBe(1);
    expect(attr(spans[0], 'neatlogs.span.kind')).toBe('TOOL');
    expect(attr(spans[0], 'neatlogs.tool.name')).toBe('get_weather');
    expect(attr(spans[0], 'output.value')).toBe('Sunny in NYC');
  });
});

// ---------------------------------------------------------------------------
// wrapAnthropic
// ---------------------------------------------------------------------------

describe('wrapAnthropic', () => {
  it('traces non-streaming messages.create', async () => {
    const fakeResponse = {
      id: 'msg_123',
      model: 'claude-sonnet-4-20250514',
      content: [
        { type: 'text', text: 'Hello there!' },
        {
          type: 'image',
          source: {
            type: 'base64',
            media_type: 'image/png',
            data: Buffer.from('anthropic-image').toString('base64'),
          },
        },
      ],
      usage: { input_tokens: 12, output_tokens: 4 },
      stop_reason: 'end_turn',
    };

    const fakeClient = {
      messages: {
        create: async () => fakeResponse,
      },
    };

    const wrapped = wrapAnthropic(fakeClient as any);
    const result = await wrapped.messages.create({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 100,
      messages: [{ role: 'user', content: 'Hi' }],
    });

    expect(result).toEqual(fakeResponse);

    const spans = getSpans();
    expect(spans.length).toBe(1);
    expect(attr(spans[0], 'neatlogs.span.kind')).toBe('LLM');
    expect(attr(spans[0], 'neatlogs.llm.provider')).toBe('anthropic');
    expect(attr(spans[0], 'neatlogs.llm.model_name')).toBe('claude-sonnet-4-20250514');
    expect(attr(spans[0], 'neatlogs.llm.output_messages.0.content')).toBe('Hello there!');
    expect(attr(spans[0], 'neatlogs.llm.output_messages.0.media.0.mime_type')).toBe('image/png');
    expect(attr(spans[0], 'neatlogs.llm.token_count.prompt')).toBe(12);
    expect(attr(spans[0], 'neatlogs.llm.token_count.completion')).toBe(4);
  });

  it('traceTool wraps a function with TOOL span', async () => {
    const search = traceToolAnthropic('web_search', async (input: { query: string }) => {
      return { results: ['result1'] };
    });

    const result = await search({ query: 'test' });
    expect(result).toEqual({ results: ['result1'] });

    const spans = getSpans();
    expect(spans.length).toBe(1);
    expect(attr(spans[0], 'neatlogs.span.kind')).toBe('TOOL');
    expect(attr(spans[0], 'neatlogs.tool.name')).toBe('web_search');
  });
});

// ---------------------------------------------------------------------------
// langchainHandler
// ---------------------------------------------------------------------------

describe('langchainHandler', () => {
  it('creates CHAIN, LLM, and TOOL spans with parent-child relationships', async () => {
    const handler = langchainHandler({ workflowName: 'test-chain' });

    await handler.handleChainStart({ name: 'RunnableSequence' }, { input: 'test' }, 'run-1');
    await handler.handleLLMStart({ kwargs: { model_name: 'gpt-4o' } }, ['Hello'], 'run-2', 'run-1');
    await handler.handleLLMEnd({ generations: [[{ message: { content: 'Hi!' } }]], llmOutput: { tokenUsage: { promptTokens: 5, completionTokens: 2, totalTokens: 7 } } }, 'run-2');
    await handler.handleToolStart({ name: 'calculator' }, '2+2', 'run-3', 'run-1');
    await handler.handleToolEnd('4', 'run-3');
    await handler.handleChainEnd({ output: 'done' }, 'run-1');

    const spans = getSpans();
    expect(spans.length).toBe(3);

    const chainSpan = spans.find(s => s.attributes['neatlogs.span.kind'] === 'CHAIN');
    const llmSpan = spans.find(s => s.attributes['neatlogs.span.kind'] === 'LLM');
    const toolSpan = spans.find(s => s.attributes['neatlogs.span.kind'] === 'TOOL');

    expect(chainSpan).toBeDefined();
    expect(llmSpan).toBeDefined();
    expect(toolSpan).toBeDefined();

    expect(attr(chainSpan!, 'neatlogs.workflow.name')).toBe('test-chain');
    expect(attr(llmSpan!, 'neatlogs.llm.model_name')).toBe('gpt-4o');
    expect(attr(llmSpan!, 'neatlogs.llm.token_count.prompt')).toBe(5);
    expect(attr(toolSpan!, 'neatlogs.tool.name')).toBe('calculator');
    expect(attr(toolSpan!, 'output.value')).toBe('4');
  });

  it('handles errors correctly', async () => {
    const handler = langchainHandler();

    await handler.handleChainStart({ name: 'test' }, {}, 'run-err');
    await handler.handleChainError(new Error('boom'), 'run-err');

    const spans = getSpans();
    expect(spans.length).toBe(1);
    expect(spans[0].status.code).toBe(2); // ERROR
  });
});

// ---------------------------------------------------------------------------
// strandsHooks
// ---------------------------------------------------------------------------

describe('strandsHooks', () => {
  it('rejects Strands because it owns the global OTel context', () => {
    expect(() => strandsHooks({ name: 'test-agent' })).toThrow(
      /cannot be isolated from other tracing SDKs/,
    );
  });
});

// ---------------------------------------------------------------------------
// openaiAgentsProcessor
// ---------------------------------------------------------------------------

describe('openaiAgentsProcessor', () => {
  it('creates WORKFLOW and AGENT spans from trace lifecycle', () => {
    const processor = openaiAgentsProcessor();

    processor.onTraceStart({ trace_id: 'trace-1', workflow_name: 'my-workflow' });
    processor.onSpanStart({ span_id: 'span-1', span_type: 'agent', agent_name: 'planner' });
    processor.onSpanEnd({ span_id: 'span-1', span_type: 'agent', output: 'done' });
    processor.onTraceEnd({ trace_id: 'trace-1' });

    const spans = getSpans();
    expect(spans.length).toBe(2);

    const workflowSpan = spans.find(s => s.attributes['neatlogs.span.kind'] === 'WORKFLOW');
    const agentSpan = spans.find(s => s.attributes['neatlogs.span.kind'] === 'AGENT');

    expect(workflowSpan).toBeDefined();
    expect(agentSpan).toBeDefined();
    expect(attr(workflowSpan!, 'neatlogs.workflow.name')).toBe('my-workflow');
    expect(attr(agentSpan!, 'neatlogs.agent.name')).toBe('planner');
  });

  it('creates LLM and TOOL spans', () => {
    const processor = openaiAgentsProcessor();

    processor.onSpanStart({ span_id: 'gen-1', span_type: 'generation', model: 'gpt-4o', messages: [{ role: 'user', content: 'hi' }] });
    processor.onSpanEnd({ span_id: 'gen-1', span_type: 'generation', output: { content: 'hello' }, usage: { input_tokens: 5, output_tokens: 2 } });

    processor.onSpanStart({ span_id: 'tool-1', span_type: 'function', name: 'search', input: '{"q":"test"}' });
    processor.onSpanEnd({ span_id: 'tool-1', span_type: 'function', output: 'results' });

    const spans = getSpans();
    expect(spans.length).toBe(2);

    const llmSpan = spans.find(s => s.attributes['neatlogs.span.kind'] === 'LLM');
    const toolSpan = spans.find(s => s.attributes['neatlogs.span.kind'] === 'TOOL');

    expect(llmSpan).toBeDefined();
    expect(attr(llmSpan!, 'neatlogs.llm.model_name')).toBe('gpt-4o');
    expect(attr(llmSpan!, 'neatlogs.llm.token_count.prompt')).toBe(5);
    expect(toolSpan).toBeDefined();
    expect(attr(toolSpan!, 'neatlogs.tool.name')).toBe('search');
  });

  it('handles the @openai/agents 0.11 Span shape (nested spanData + _response)', () => {
    const processor = openaiAgentsProcessor();

    // Trace + agent + response (LLM) + function (tool), as the real SDK passes them.
    processor.onTraceStart({ traceId: 'trace-x', name: 'Agent workflow' });
    processor.onSpanStart({ spanId: 'a1', traceId: 'trace-x', spanData: { type: 'agent', name: 'Weather Assistant', tools: ['get_weather'] } });
    processor.onSpanStart({ spanId: 'r1', traceId: 'trace-x', spanData: { type: 'response' } });
    processor.onSpanEnd({ spanId: 'r1', traceId: 'trace-x', spanData: { type: 'response', _input: [
      { role: 'system', content: 'Be concise' },
      { role: 'user', content: 'Where is order 42?' },
    ], _response: {
      model: 'gpt-4o-mini-2024-07-18',
      usage: { input_tokens: 57, output_tokens: 15, total_tokens: 72 },
      output: [{ type: 'message', role: 'assistant', content: [{ type: 'output_text', text: 'Sunny' }] }],
    } } });
    processor.onSpanStart({ spanId: 't1', traceId: 'trace-x', spanData: { type: 'function', name: 'get_weather', input: '{"city":"Paris"}' } });
    processor.onSpanEnd({ spanId: 't1', traceId: 'trace-x', spanData: { type: 'function', name: 'get_weather', output: '18C' } });
    processor.onSpanEnd({ spanId: 'a1', traceId: 'trace-x', spanData: { type: 'agent', name: 'Weather Assistant' } });
    processor.onTraceEnd({ traceId: 'trace-x' });

    const spans = getSpans();
    const kinds = spans.map(s => s.attributes['neatlogs.span.kind']).sort();
    expect(kinds).toEqual(['AGENT', 'LLM', 'TOOL', 'WORKFLOW']);

    const llm = spans.find(s => s.attributes['neatlogs.span.kind'] === 'LLM')!;
    expect(attr(llm, 'neatlogs.llm.model_name')).toBe('gpt-4o-mini-2024-07-18');
    expect(attr(llm, 'neatlogs.llm.token_count.total')).toBe(72);
    expect(attr(llm, 'neatlogs.llm.output_messages.0.content')).toBe('Sunny');

    const workflow = spans.find(s => s.attributes['neatlogs.span.kind'] === 'WORKFLOW')!;
    expect(attr(workflow, 'input.value')).toBe('Where is order 42?');
    expect(attr(workflow, 'output.value')).toBe('Sunny');
    expect(attr(llm, 'neatlogs.llm.input_messages.0.role')).toBe('system');
    expect(attr(llm, 'neatlogs.llm.input_messages.1.role')).toBe('user');
    expect(attr(llm, 'neatlogs.llm.input_messages.1.content')).toBe('Where is order 42?');

    const tool = spans.find(s => s.attributes['neatlogs.span.kind'] === 'TOOL')!;
    expect(attr(tool, 'neatlogs.tool.name')).toBe('get_weather');

    const agent = spans.find(s => s.attributes['neatlogs.span.kind'] === 'AGENT')!;
    expect(attr(agent, 'neatlogs.agent.name')).toBe('Weather Assistant');
  });

  it('shutdown cleans up pending spans', () => {
    const processor = openaiAgentsProcessor();
    processor.onSpanStart({ span_id: 'orphan', span_type: 'agent', agent_name: 'x' });
    processor.shutdown();

    const spans = getSpans();
    expect(spans.length).toBe(1);
    expect(spans[0].status.code).toBe(2); // ERROR
  });
  it('keeps data URLs in text parts out of root input and output', () => {
    const processor = openaiAgentsProcessor();
    const data = Buffer.alloc(120_000, 0x42).toString('base64');
    const url = `data:image/png;base64,${data}`;
    processor.onTraceStart({ traceId: 'trace-t', name: 'Agent workflow' });
    processor.onSpanStart({ spanId: 'r2', traceId: 'trace-t', spanData: { type: 'response' } });
    processor.onSpanEnd({ spanId: 'r2', traceId: 'trace-t', spanData: { type: 'response', _input: [
      { role: 'user', content: [{ type: 'input_text', text: url }] },
    ], _response: {
      model: 'gpt-4o-mini',
      output: [{ type: 'message', role: 'assistant', content: [{ type: 'output_text', text: url }] }],
    } } });
    processor.onTraceEnd({ traceId: 'trace-t' });

    const spans = exporter.getFinishedSpans();
    expect(JSON.stringify(spans.map(s => s.attributes))).not.toContain(data.slice(0, 256));
    for (const span of spans) discardPendingMedia(span as object);
  });

  it('captures only new input turns per response span', () => {
    const processor = openaiAgentsProcessor();
    processor.onTraceStart({ traceId: 'trace-long', name: 'Agent workflow' });
    const history: any[] = [];
    for (let turn = 0; turn < 40; turn++) {
      history.push({ role: 'user', content: `question ${turn} ` + 'q'.repeat(2000) });
      const id = `r${turn}`;
      processor.onSpanStart({ spanId: id, traceId: 'trace-long', spanData: { type: 'response' } });
      processor.onSpanEnd({ spanId: id, traceId: 'trace-long', spanData: { type: 'response', _input: history.slice(), _response: {
        model: 'gpt-4o-mini',
        output: [{ type: 'message', role: 'assistant', content: [{ type: 'output_text', text: `answer ${turn}` }] }],
        usage: { input_tokens: 10, output_tokens: 2 },
      } } });
      history.push({ role: 'assistant', content: `answer ${turn}` });
    }
    processor.onTraceEnd({ traceId: 'trace-long' });

    const llm = exporter.getFinishedSpans().filter(s => s.attributes['neatlogs.span.kind'] === 'LLM');
    expect(llm.length).toBe(40);
    for (const span of llm) {
      const chars = Object.entries(span.attributes)
        .filter(([k]) => /^neatlogs\.llm\.input_messages\.\d+\.content$/.test(k))
        .reduce((n, [, v]) => n + String(v).length, 0);
      expect(chars).toBeLessThan(2 * 2100 + 100);
      // Usage and output are still written however long the history is.
      expect(attr(span, 'neatlogs.llm.token_count.prompt')).toBe(10);
      expect(attr(span, 'neatlogs.llm.output_messages.0.content')).toMatch(/^answer /);
    }
    // The last span holds only the two turns added since the previous span, not the 77 earlier ones.
    const last = llm[llm.length - 1];
    const inputKeys = Object.keys(last.attributes).filter(k => /^neatlogs\.llm\.input_messages\.\d+\.content$/.test(k));
    expect(inputKeys).toEqual(['neatlogs.llm.input_messages.77.content', 'neatlogs.llm.input_messages.78.content']);
    // Total captured across the trace is about one copy of the history, not one per span.
    const total = llm.reduce((n, s) => n + Object.entries(s.attributes)
      .filter(([k]) => /^neatlogs\.llm\.input_messages\.\d+\.content$/.test(k))
      .reduce((m, [, v]) => m + String(v).length, 0), 0);
    expect(total).toBeLessThan(40 * 2100 + 40 * 20);
  });

  function runResponses(traceId: string, inputs: any[][]) {
    const processor = openaiAgentsProcessor();
    processor.onTraceStart({ traceId, name: 'Agent workflow' });
    inputs.forEach((input, n) => {
      const spanId = `${traceId}-r${n}`;
      processor.onSpanStart({ spanId, traceId, spanData: { type: 'response' } });
      processor.onSpanEnd({ spanId, traceId, spanData: { type: 'response', _input: input, _response: {
        model: 'gpt-4o-mini',
        output: [{ type: 'message', role: 'assistant', content: [{ type: 'output_text', text: `answer ${n}` }] }],
      } } });
    });
    processor.onTraceEnd({ traceId });
    return exporter.getFinishedSpans().filter(s => s.attributes['neatlogs.span.kind'] === 'LLM');
  }

  it('captures a same-text message whose media differs from an earlier response', () => {
    const image = (url: string) => [{ role: 'user', content: [
      { type: 'input_text', text: 'What is in this image?' },
      { type: 'input_image', image_url: url },
    ] }];
    const llm = runResponses('trace-media-id', [
      image('https://example.com/a.png'),
      image('https://example.com/b.png'),
    ]);
    expect(llm.length).toBe(2);
    expect(attr(llm[0], 'neatlogs.llm.input_messages.0.media.0.reference')).toBe('https://example.com/a.png');
    expect(attr(llm[1], 'neatlogs.llm.input_messages.0.media.0.reference')).toBe('https://example.com/b.png');
  });

  it('still skips a repeated message with identical text and media', () => {
    const first = [{ role: 'user', content: [
      { type: 'input_text', text: 'look' },
      { type: 'input_image', image_url: 'https://example.com/a.png' },
    ] }];
    const llm = runResponses('trace-media-same', [first, [...first, { role: 'user', content: 'and now?' }]]);
    expect(attr(llm[1], 'neatlogs.llm.input_messages.0.content')).toBeUndefined();
    expect(attr(llm[1], 'neatlogs.llm.input_messages.1.content')).toBe('and now?');
  });

  it('keeps a long first message whole and the short message after it', () => {
    const long = 'x'.repeat(21000);
    const history = [{ role: 'user', content: long }, { role: 'user', content: 'short follow up' }];
    const llm = runResponses('trace-long-first', [history, history]);
    expect(attr(llm[0], 'neatlogs.llm.input_messages.0.content')).toBe(long);
    expect(attr(llm[0], 'neatlogs.llm.input_messages.1.content')).toBe('short follow up');
    expect(attr(llm[1], 'neatlogs.llm.input_messages.0.content')).toBeUndefined();
    expect(attr(llm[1], 'neatlogs.llm.input_messages.1.content')).toBeUndefined();
  });

  it('writes every message of a very long input on the only response span', () => {
    // No later response span exists to pick up anything left out.
    const big = (label: string) => `${label} ` + 'y'.repeat(300 * 1024);
    const history = [
      { role: 'user', content: big('one') },
      { role: 'user', content: big('two') },
      { role: 'user', content: 'tail' },
    ];
    const llm = runResponses('trace-terminal', [history]);
    expect(llm.length).toBe(1);
    expect(attr(llm[0], 'neatlogs.llm.input_messages.0.content')).toBe(history[0].content);
    expect(attr(llm[0], 'neatlogs.llm.input_messages.1.content')).toBe(history[1].content);
    expect(attr(llm[0], 'neatlogs.llm.input_messages.2.content')).toBe('tail');
  });

  it('keeps text part boundaries distinct when delimiter text matches', () => {
    const llm = runResponses('trace-part-boundaries', [
      [{ role: 'user', content: [{ type: 'input_text', text: 'a: input_text:b' }] }],
      [{ role: 'user', content: [
        { type: 'input_text', text: 'a' },
        { type: 'input_text', text: 'b' },
      ] }],
    ]);
    expect(attr(llm[0], 'neatlogs.llm.input_messages.0.content')).toBe('a: input_text:b');
    expect(attr(llm[1], 'neatlogs.llm.input_messages.0.content')).toBe('ab');
  });

  it('captures same-length but different input histories on later response spans', () => {
    // Regression: a trace-wide message count treated a same-length list as an
    // already captured prefix, so a distinct one-message _input lost its input.
    const processor = openaiAgentsProcessor();
    processor.onTraceStart({ traceId: 'trace-sl', name: 'Agent workflow' });
    processor.onSpanStart({ spanId: 'r1', traceId: 'trace-sl', spanData: { type: 'response' } });
    processor.onSpanEnd({ spanId: 'r1', traceId: 'trace-sl', spanData: { type: 'response', _input: [
      { role: 'user', content: 'first prompt' },
    ], _response: {
      model: 'gpt-4o-mini',
      output: [{ type: 'message', role: 'assistant', content: [{ type: 'output_text', text: 'first answer' }] }],
    } } });
    processor.onSpanStart({ spanId: 'r2', traceId: 'trace-sl', spanData: { type: 'response' } });
    processor.onSpanEnd({ spanId: 'r2', traceId: 'trace-sl', spanData: { type: 'response', _input: [
      { role: 'user', content: 'a different prompt' },
    ], _response: {
      model: 'gpt-4o-mini',
      output: [{ type: 'message', role: 'assistant', content: [{ type: 'output_text', text: 'second answer' }] }],
    } } });
    processor.onTraceEnd({ traceId: 'trace-sl' });

    const llm = exporter.getFinishedSpans().filter(s => s.attributes['neatlogs.span.kind'] === 'LLM');
    expect(llm.length).toBe(2);
    expect(attr(llm[0], 'neatlogs.llm.input_messages.0.content')).toBe('first prompt');
    expect(attr(llm[1], 'neatlogs.llm.input_messages.0.content')).toBe('a different prompt');
  });

  it('captures input on interleaved parallel branches sharing one trace', () => {
    // Regression: two branches interleave response spans with diverging
    // histories; only a verified shared prefix may be skipped, so neither
    // branch loses its turns.
    const processor = openaiAgentsProcessor();
    const respond = (id: string, input: any[], answer: string) => {
      processor.onSpanStart({ spanId: id, traceId: 'trace-cb', spanData: { type: 'response' } });
      processor.onSpanEnd({ spanId: id, traceId: 'trace-cb', spanData: { type: 'response', _input: input, _response: {
        model: 'gpt-4o-mini',
        output: [{ type: 'message', role: 'assistant', content: [{ type: 'output_text', text: answer }] }],
      } } });
    };
    const shared = { role: 'user', content: 'shared start' };
    processor.onTraceStart({ traceId: 'trace-cb', name: 'Agent workflow' });
    respond('a1', [shared, { role: 'user', content: 'branch a turn 1' }], 'a1');
    respond('b1', [shared, { role: 'user', content: 'branch b turn 1' }], 'b1');
    respond('a2', [shared, { role: 'user', content: 'branch a turn 1' }, { role: 'assistant', content: 'a1' }, { role: 'user', content: 'branch a turn 2' }], 'a2');
    processor.onTraceEnd({ traceId: 'trace-cb' });

    const llm = exporter.getFinishedSpans().filter(s => s.attributes['neatlogs.span.kind'] === 'LLM');
    expect(llm.length).toBe(3);
    expect(attr(llm[0], 'neatlogs.llm.input_messages.0.content')).toBe('shared start');
    expect(attr(llm[0], 'neatlogs.llm.input_messages.1.content')).toBe('branch a turn 1');
    // The b branch keeps its own turn even though its list matched a's length.
    expect(attr(llm[1], 'neatlogs.llm.input_messages.1.content')).toBe('branch b turn 1');
    // The continued a branch still lands its newest turn.
    expect(attr(llm[2], 'neatlogs.llm.input_messages.3.content')).toBe('branch a turn 2');
  });

  it('keeps inline media out of response and root text attributes', () => {
    const processor = openaiAgentsProcessor();
    const data = Buffer.alloc(120_000, 0x41).toString('base64');
    processor.onTraceStart({ traceId: 'trace-m', name: 'Agent workflow' });
    processor.onSpanStart({ spanId: 'r1', traceId: 'trace-m', spanData: { type: 'response' } });
    processor.onSpanEnd({ spanId: 'r1', traceId: 'trace-m', spanData: { type: 'response', _input: [
      { role: 'user', content: [
        { type: 'input_text', text: 'What is in this image?' },
        { type: 'input_image', image_url: `data:image/png;base64,${data}` },
      ] },
    ], _response: {
      model: 'gpt-4o-mini',
      output: [{ type: 'message', role: 'assistant', content: [{ type: 'output_text', text: 'A cat' }] }],
    } } });
    processor.onTraceEnd({ traceId: 'trace-m' });

    const spans = exporter.getFinishedSpans();
    expect(JSON.stringify(spans.map(s => s.attributes))).not.toContain(data.slice(0, 256));
    const llm = spans.find(s => s.attributes['neatlogs.span.kind'] === 'LLM')!;
    const workflow = spans.find(s => s.attributes['neatlogs.span.kind'] === 'WORKFLOW')!;
    expect(attr(llm, 'neatlogs.llm.input_messages.0.content')).toBe('What is in this image?');
    expect(attr(llm, 'neatlogs.llm.input_messages.0.media.0.mime_type')).toBe('image/png');
    expect(attr(workflow, 'input.value')).toBe('What is in this image?');
    for (const span of spans) discardPendingMedia(span as object);
  });
});

// ---------------------------------------------------------------------------
// wrapMastra
// ---------------------------------------------------------------------------

describe('wrapMastra', () => {
  it('wraps Agent.generate() with AGENT span', async () => {
    const fakeAgent = {
      name: 'planner',
      model: { modelId: 'gpt-4o' },
      generate: async (input: string) => ({
        text: 'Plan: do things',
        usage: {
          promptTokens: 10,
          completionTokens: 5,
          totalTokens: 15,
          reasoningTokens: 2,
          cachedInputTokens: 7,
          cacheCreationInputTokens: 3,
        },
        finishReason: 'stop',
        toolCalls: [],
      }),
    };

    const wrapped = wrapMastra(fakeAgent as any);
    const result = await (wrapped as any).generate('Create a plan');

    expect(result.text).toBe('Plan: do things');

    const spans = getSpans();
    expect(spans.length).toBe(1);
    expect(attr(spans[0], 'neatlogs.span.kind')).toBe('AGENT');
    expect(attr(spans[0], 'neatlogs.agent.name')).toBe('planner');
    expect(attr(spans[0], 'neatlogs.llm.model_name')).toBe('gpt-4o');
    expect(attr(spans[0], 'neatlogs.llm.output_messages.0.content')).toBe('Plan: do things');
    expect(attr(spans[0], 'output.value')).toBe('Plan: do things');
    expect(attr(spans[0], 'neatlogs.llm.token_count.prompt')).toBe(10);
    expect(attr(spans[0], 'neatlogs.llm.token_count.reasoning')).toBe(2);
    expect(attr(spans[0], 'neatlogs.llm.token_count.cache_read')).toBe(7);
    expect(attr(spans[0], 'neatlogs.llm.token_count.cache_write')).toBe(3);
  });

  it('records canonical AGENT and LLM output for a provider stream', async () => {
    const model = {
      modelId: 'stream-model',
      provider: 'test',
      doStream: async () => ({
        stream: new ReadableStream({
          start(controller) {
            controller.enqueue({ type: 'text-delta', delta: 'Hel' });
            controller.enqueue({ type: 'text-delta', delta: 'lo' });
            controller.enqueue({
              type: 'finish',
              finishReason: 'stop',
              usage: {
                inputTokens: { total: 4, cacheRead: 3, cacheWrite: 1 },
                outputTokens: { total: 2, reasoning: 1 },
              },
            });
            controller.close();
          },
        }),
      }),
    };
    const agent = wrapMastra({
      name: 'streaming-agent',
      getLLM: async () => ({ getModel: () => model }),
      stream: async function () {
        const llm = await this.getLLM();
        const result = await llm.getModel().doStream({
          prompt: [{ role: 'user', content: 'Say hello' }],
        });
        const reader = result.stream.getReader();
        let text = '';
        while (true) {
          const next = await reader.read();
          if (next.done) break;
          if (next.value.type === 'text-delta') text += next.value.delta;
        }
        return {
          text: Promise.resolve(text),
          usage: Promise.resolve({
            inputTokens: { total: 4, cacheRead: 3, cacheWrite: 1 },
            outputTokens: { total: 2, reasoning: 1 },
          }),
          finishReason: Promise.resolve('stop'),
          fullStream: new ReadableStream({ start(controller) { controller.close(); } }),
        };
      },
    });

    const output = await agent.stream('Say hello');
    expect(await output.text).toBe('Hello');
    await new Promise((resolve) => setTimeout(resolve, 0));

    const spans = getSpans();
    const agentSpan = spans.find((span) => span.name === 'mastra.agent.streaming-agent')!;
    const llmSpan = spans.find((span) => span.name === 'mastra.llm.stream-model.doStream')!;
    expect(attr(agentSpan, 'output.value')).toBe('Hello');
    expect(attr(llmSpan, 'output.value')).toBe('Hello');
    expect(attr(llmSpan, 'neatlogs.llm.output_messages.0.content')).toBe('Hello');
    expect(attr(llmSpan, 'neatlogs.llm.token_count.total')).toBe(6);
    expect(attr(llmSpan, 'neatlogs.llm.token_count.reasoning')).toBe(1);
    expect(attr(llmSpan, 'neatlogs.llm.token_count.cache_read')).toBe(3);
    expect(attr(llmSpan, 'neatlogs.llm.token_count.cache_write')).toBe(1);
    expect(llmSpan.parentSpanId).toBe(agentSpan.spanContext().spanId);
  });

  it('wraps Workflow.createRun().start() with WORKFLOW span', async () => {
    const fakeWorkflow = {
      name: 'health-check',
      createRun: async () => ({
        start: async (opts: any) => ({
          status: 'completed',
          result: { score: 85 },
        }),
      }),
    };

    const wrapped = wrapMastra(fakeWorkflow as any);
    const run = await (wrapped as any).createRun();
    const result = await run.start({ inputData: { accountId: '123' } });

    expect(result.status).toBe('completed');

    const spans = getSpans();
    expect(spans.length).toBe(1);
    expect(attr(spans[0], 'neatlogs.span.kind')).toBe('WORKFLOW');
    expect(attr(spans[0], 'neatlogs.workflow.name')).toBe('health-check');
    // status is not a canonical neatlogs key — it is folded into neatlogs.metadata
    expect(attr(spans[0], 'neatlogs.metadata')).toContain('completed');
    expect(attr(spans[0], 'input.value')).toContain('accountId');
    expect(attr(spans[0], 'output.value')).toContain('score');
  });

  it('records input and output for document chunk processors', async () => {
    class MDocument {
      getDocs() {
        return [];
      }

      async chunk(options: unknown) {
        return [{ text: 'one' }, { text: 'two' }, options];
      }
    }

    const document = wrapMastra(new MDocument());
    await document.chunk({ strategy: 'recursive' });

    const span = getSpans().find((candidate) => candidate.name === 'mastra.document.chunk')!;
    expect(attr(span, 'input.value')).toContain('recursive');
    expect(attr(span, 'output.value')).toContain('one');
    expect(attr(span, 'neatlogs.db.documents_count')).toBe(3);
  });

  it('records meaningful output for void-returning memory writes', async () => {
    class Memory {
      async saveMessages() {
        return undefined;
      }
    }

    const memory = wrapMastra(new Memory());
    await memory.saveMessages({
      messages: [{ role: 'user', content: 'one' }, { role: 'assistant', content: 'two' }],
    });

    const span = getSpans().find((candidate) => candidate.name === 'mastra.memory.saveMessages')!;
    expect(attr(span, 'input.value')).toContain('messages');
    expect(attr(span, 'output.value')).toBe('{"completed":true,"messageCount":2}');
  });

  it('is idempotent', async () => {
    const fakeAgent = { name: 'a', generate: async () => ({ text: 'ok' }) };
    wrapMastra(fakeAgent as any);
    const first = fakeAgent.generate;
    wrapMastra(fakeAgent as any);
    expect(fakeAgent.generate).toBe(first);
  });
});
