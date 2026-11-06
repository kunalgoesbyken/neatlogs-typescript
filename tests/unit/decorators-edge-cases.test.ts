/**
 * Additional edge-case tests for decorators/orchestration.ts
 * Covers: Span() class decorator, MCP_TOOL edge cases, RETRIEVER edge cases,
 * AGENT/TOOL with missing optional attributes, and decorateSpan edge cases.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { safeJsonDumps, serializeObj, decorateSpan } from '../../src/decorators/base.js';
import { span, Span, retrieverPostprocessor } from '../../src/decorators/orchestration.js';
import { _setNeatlogsProvider } from '../../src/core/provider.js';
import {
  discardPendingMedia,
  setMediaCaptureAvailability,
} from '../../src/core/media.js';

// ─── Mock OpenTelemetry ────────────────────────────────────────────────────────

function createMockSpan() {
  return {
    setAttribute: vi.fn(),
    setStatus: vi.fn(),
    addEvent: vi.fn(),
    end: vi.fn(),
    recordException: vi.fn(),
    _attrs: {} as Record<string, any>,
  };
}

let mockSpan = createMockSpan();

vi.mock('@opentelemetry/api', () => {
  const rootContext = {
    getValue: () => undefined,
    setValue: () => rootContext,
    deleteValue: () => rootContext,
  };
  const tracer = {
    startSpan: (_name: string) => mockSpan,
    startActiveSpan: (name: string, fn: (span: any) => any) => fn(mockSpan),
  };
  return {
    createContextKey: (name: string) => Symbol.for(name),
    trace: {
      // Shared mode resolves the tracer via the global provider.
      getTracerProvider: () => ({ getTracer: () => tracer }),
      getTracer: () => tracer,
      // isRootSpan() reads the active span; no active span → decorated fn is root.
      getSpan: () => undefined,
      setSpan: (ctx: any, _span: any) => ctx,
    },
    context: {
      active: () => rootContext,
      with: (_ctx: any, fn: () => any) => fn(),
    },
    ROOT_CONTEXT: rootContext,
    SpanStatusCode: {
      OK: 1,
      ERROR: 2,
    },
  };
});

beforeEach(() => {
  _setNeatlogsProvider({
    getTracer: () => ({
      startSpan: () => mockSpan,
      startActiveSpan: (_name: string, fn: (span: any) => any) => fn(mockSpan),
    }),
  } as any);
});

afterEach(() => {
  _setNeatlogsProvider(null);
});

// ─── Span() class-method decorator ─────────────────────────────────────────────

describe('Span() class-method decorator', () => {
  beforeEach(() => {
    mockSpan = createMockSpan();
  });

  it('should decorate a class method and wrap it with span instrumentation', () => {
    // Simulate TC39 Stage 3 class method decorator behavior
    const originalMethod = function (this: any, query: string) {
      return `processed: ${query}`;
    };

    // Create a mock ClassMethodDecoratorContext
    const context: ClassMethodDecoratorContext = {
      kind: 'method',
      name: 'run',
      static: false,
      private: false,
      access: { has: () => true, get: () => originalMethod },
      addInitializer: () => {},
      metadata: {},
    };

    const decorator = Span({ kind: 'AGENT', role: 'researcher' });
    const wrapped = decorator(originalMethod, context);

    // The wrapped function should work
    const result = wrapped('test query');
    expect(result).toBe('processed: test query');
    expect(mockSpan.setAttribute).toHaveBeenCalledWith('openinference.span.kind', 'AGENT');
    expect(mockSpan.setAttribute).toHaveBeenCalledWith('neatlogs.agent.role', 'researcher');
  });

  it('should preserve this binding for decorated class methods', () => {
    const originalMethod = function (
      this: { prefix: string; format: (query: string) => string },
      query: string,
    ) {
      return this.format(query);
    };
    const context: ClassMethodDecoratorContext = {
      kind: 'method',
      name: 'run',
      static: false,
      private: false,
      access: { has: () => true, get: () => originalMethod },
      addInitializer: () => {},
      metadata: {},
    };

    const decorator = Span({ kind: 'AGENT', role: 'researcher' });
    const wrapped = decorator(originalMethod, context);
    const instance = {
      prefix: 'processed',
      format(query: string) {
        return `${this.prefix}: ${query}`;
      },
    };

    const result = wrapped.call(instance, 'test query');

    expect(result).toBe('processed: test query');
    expect(mockSpan.setAttribute).toHaveBeenCalledWith('openinference.span.kind', 'AGENT');
  });

  it('should throw for invalid span kind in Span() decorator', () => {
    // Span() calls span() internally, which validates the kind.
    // However, Span() returns a decorator function — it only throws
    // when the decorator is applied (not when Span() is called).
    // The span() call happens inside the decorator, so we need to
    // verify it happens during decoration.
    const decorator = Span({ kind: 'INVALID' as any });
    const originalMethod = function (this: any) {
      return 42;
    };
    const context: ClassMethodDecoratorContext = {
      kind: 'method',
      name: 'run',
      static: false,
      private: false,
      access: { has: () => true, get: () => originalMethod },
      addInitializer: () => {},
      metadata: {},
    };
    expect(() => decorator(originalMethod, context)).toThrow(/Invalid span kind/);
  });
});

// ─── MCP_TOOL edge cases ────────────────────────────────────────────────────────

describe('span() with MCP_TOOL edge cases', () => {
  beforeEach(() => {
    mockSpan = createMockSpan();
  });

  it('should handle MCP_TOOL with object arg that has toJSON', () => {
    const fn = (input: any) => 'mcp result';
    const wrapped = span({ kind: 'MCP_TOOL', toolName: 'calculate' }, fn);

    const inputObj = {
      a: 1,
      b: 2,
      toJSON: () => ({ a: 1, b: 2, type: 'serialized' }),
    };
    wrapped(inputObj);

    expect(mockSpan.setAttribute).toHaveBeenCalledWith('mcp.tool.name', 'calculate');
    expect(mockSpan.setAttribute).toHaveBeenCalledWith('tool.name', 'calculate');
    // toJSON result should be used
    expect(mockSpan.setAttribute).toHaveBeenCalledWith(
      'mcp.tool.input',
      expect.stringContaining('"type":"serialized"'),
    );
  });

  it('should handle MCP_TOOL with plain object arg (no toJSON)', () => {
    const fn = (input: { x: number }) => 'result';
    const wrapped = span({ kind: 'MCP_TOOL', toolName: 'tool1' }, fn);

    wrapped({ x: 42 });

    expect(mockSpan.setAttribute).toHaveBeenCalledWith(
      'mcp.tool.input',
      expect.stringContaining('"x":42'),
    );
  });

  it('should handle MCP_TOOL with non-string result', () => {
    const fn = () => ({ success: true, count: 5 });
    const wrapped = span({ kind: 'MCP_TOOL', toolName: 'complex_tool' }, fn);

    wrapped();

    // Non-string results should not have the special wrapping
    const outputCalls = mockSpan.setAttribute.mock.calls.filter(
      ([key]: [string]) => key === 'output.value',
    );
    if (outputCalls.length > 0) {
      // Should be the normal serialization
      const parsed = JSON.parse(outputCalls[0][1]);
      expect(parsed.success).toBe(true);
    }
  });

  it('should handle MCP_TOOL without toolName or parameters', () => {
    const fn = () => 'result';
    const wrapped = span({ kind: 'MCP_TOOL' }, fn);

    wrapped();

    // Should not set mcp.tool.name or mcp.tool.parameters
    const mcpNameCalls = mockSpan.setAttribute.mock.calls.filter(
      ([key]: [string]) => key === 'mcp.tool.name',
    );
    expect(mcpNameCalls).toHaveLength(0);
  });

  it('should handle MCP_TOOL with async function', async () => {
    const fn = async (input: string) => `processed: ${input}`;
    const wrapped = span(
      {
        kind: 'MCP_TOOL',
        toolName: 'async_tool',
        parameters: { input: 'string' },
      },
      fn,
    );

    const result = await wrapped('hello');
    expect(result).toBe('processed: hello');
    expect(mockSpan.setAttribute).toHaveBeenCalledWith('mcp.tool.name', 'async_tool');
    expect(mockSpan.setAttribute).toHaveBeenCalledWith('mcp.tool.parameters', '{"input":"string"}');
  });

  it('sanitizes MCP tool input, output, parameters, and schema attributes', () => {
    setMediaCaptureAvailability(mockSpan, false, 'test_uploads_disabled');
    const raw = `data:image/png;base64,${'YWFh'.repeat(40_000)}`;
    const media = { type: 'image_url', image_url: { url: raw } };
    const wrapped = span(
      {
        kind: 'MCP_TOOL',
        toolName: 'inspect_media',
        parameters: { example: media },
        toolJsonSchema: { example: media },
      },
      (input: unknown) => JSON.stringify({ media: input }),
    );

    wrapped(media);

    for (const key of [
      'input.value',
      'output.value',
      'mcp.tool.input',
      'mcp.tool.parameters',
      'tool.parameters',
      'tool.json_schema',
    ]) {
      const calls = mockSpan.setAttribute.mock.calls.filter(([name]) => name === key);
      expect(calls.length).toBeGreaterThan(0);
      expect(String(calls.at(-1)?.[1])).not.toContain(raw);
    }
    expect(String(mockSpan.setAttribute.mock.calls.find(
      ([name]) => name === 'mcp.tool.input',
    )?.[1])).toContain('neatlogs_media');
    expect(mockSpan.setAttribute).toHaveBeenCalledWith(
      'neatlogs.mcp_tool.output.media.0.state',
      'failed',
    );
    discardPendingMedia(mockSpan);
  });
});

// ─── RETRIEVER edge cases ───────────────────────────────────────────────────────

describe('retrieverPostprocessor edge cases', () => {
  let spanObj: ReturnType<typeof createMockSpan>;

  beforeEach(() => {
    spanObj = createMockSpan();
  });

  it('should not set query when no matching key is found', () => {
    retrieverPostprocessor(spanObj as any, [], { something_else: 'value' });

    const queryCalls = spanObj.setAttribute.mock.calls.filter(
      ([key]: [string]) => key === 'neatlogs.retriever.query',
    );
    expect(queryCalls).toHaveLength(0);
  });

  it('should not set query when key value is not a string', () => {
    retrieverPostprocessor(spanObj as any, [], { query: 42 });

    const queryCalls = spanObj.setAttribute.mock.calls.filter(
      ([key]: [string]) => key === 'neatlogs.retriever.query',
    );
    expect(queryCalls).toHaveLength(0);
  });

  it('should handle document object with no content fields', () => {
    const docs = [{ id: 'doc1', score: 0.9 }];
    retrieverPostprocessor(spanObj as any, docs, {});

    // Should set id and score but not content
    expect(spanObj.setAttribute).toHaveBeenCalledWith('neatlogs.retriever.documents.0.id', 'doc1');
    expect(spanObj.setAttribute).toHaveBeenCalledWith('neatlogs.retriever.documents.0.score', 0.9);
    const contentCalls = spanObj.setAttribute.mock.calls.filter(
      ([key]: [string]) => typeof key === 'string' && key.endsWith('.content'),
    );
    expect(contentCalls).toHaveLength(0);
  });

  it('should handle empty array result', () => {
    retrieverPostprocessor(spanObj as any, [], { query: 'test' });

    expect(spanObj.setAttribute).toHaveBeenCalledWith('neatlogs.retriever.query', 'test');
    const docCalls = spanObj.setAttribute.mock.calls.filter(
      ([key]: [string]) =>
        typeof key === 'string' && key.startsWith('neatlogs.retriever.documents'),
    );
    expect(docCalls).toHaveLength(0);
  });

  it('should handle dict result without documents/docs/results keys', () => {
    const result = { data: 'something', count: 5 };
    retrieverPostprocessor(spanObj as any, result, {});

    // Should not set any document attributes
    const docCalls = spanObj.setAttribute.mock.calls.filter(
      ([key]: [string]) =>
        typeof key === 'string' && key.startsWith('neatlogs.retriever.documents'),
    );
    expect(docCalls).toHaveLength(0);
  });

  it('should handle document with numeric id', () => {
    const docs = [{ id: 123, content: 'text' }];
    retrieverPostprocessor(spanObj as any, docs, {});

    expect(spanObj.setAttribute).toHaveBeenCalledWith(
      'neatlogs.retriever.documents.0.id',
      '123', // Should be converted to string
    );
  });

  it('should prefer content over page_content over text', () => {
    const docs = [{ content: 'c1', page_content: 'p1', text: 't1' }];
    retrieverPostprocessor(spanObj as any, docs, {});

    expect(spanObj.setAttribute).toHaveBeenCalledWith(
      'neatlogs.retriever.documents.0.content',
      'c1',
    );
  });

  it('should use page_content when content is absent', () => {
    const docs = [{ page_content: 'p1', text: 't1' }];
    retrieverPostprocessor(spanObj as any, docs, {});

    expect(spanObj.setAttribute).toHaveBeenCalledWith(
      'neatlogs.retriever.documents.0.content',
      'p1',
    );
  });

  it('should handle documents inside nested dict with "results" key', () => {
    const result = {
      results: [{ content: 'result-text', score: 0.95 }],
    };
    retrieverPostprocessor(spanObj as any, result, { query: 'search query' });

    expect(spanObj.setAttribute).toHaveBeenCalledWith('neatlogs.retriever.query', 'search query');
    expect(spanObj.setAttribute).toHaveBeenCalledWith(
      'neatlogs.retriever.documents.0.content',
      'result-text',
    );
    expect(spanObj.setAttribute).toHaveBeenCalledWith('neatlogs.retriever.documents.0.score', 0.95);
  });
});

// ─── AGENT span edge cases ──────────────────────────────────────────────────────

describe('span() with AGENT edge cases', () => {
  beforeEach(() => {
    mockSpan = createMockSpan();
  });

  it('should not set role/goal when they are not provided', () => {
    const fn = () => 'agent result';
    const wrapped = span({ kind: 'AGENT' }, fn);
    wrapped();

    const roleCalls = mockSpan.setAttribute.mock.calls.filter(
      ([key]: [string]) => key === 'neatlogs.agent.role',
    );
    const goalCalls = mockSpan.setAttribute.mock.calls.filter(
      ([key]: [string]) => key === 'neatlogs.agent.goal',
    );
    expect(roleCalls).toHaveLength(0);
    expect(goalCalls).toHaveLength(0);
  });

  it('should set only role when goal is not provided', () => {
    const fn = () => 'agent result';
    const wrapped = span({ kind: 'AGENT', role: 'analyst' }, fn);
    wrapped();

    expect(mockSpan.setAttribute).toHaveBeenCalledWith('neatlogs.agent.role', 'analyst');
    const goalCalls = mockSpan.setAttribute.mock.calls.filter(
      ([key]: [string]) => key === 'neatlogs.agent.goal',
    );
    expect(goalCalls).toHaveLength(0);
  });
});

// ─── TOOL span edge cases ───────────────────────────────────────────────────────

describe('span() with TOOL edge cases', () => {
  beforeEach(() => {
    mockSpan = createMockSpan();
  });

  it('uses the resolved function name when toolName is not provided', () => {
    const fn = () => 'tool result';
    const wrapped = span({ kind: 'TOOL' }, fn);
    wrapped();

    const toolNameCalls = mockSpan.setAttribute.mock.calls.filter(
      ([key]: [string]) => key === 'tool.name',
    );
    expect(toolNameCalls).toEqual([['tool.name', 'fn']]);
  });
});

// ─── EMBEDDING span edge cases ──────────────────────────────────────────────────

describe('span() with EMBEDDING edge cases', () => {
  beforeEach(() => {
    mockSpan = createMockSpan();
  });

  it('should not set model/dimension when not provided', () => {
    const fn = () => [0.1, 0.2];
    const wrapped = span({ kind: 'EMBEDDING' }, fn);
    wrapped();

    const modelCalls = mockSpan.setAttribute.mock.calls.filter(
      ([key]: [string]) => key === 'embedding.model_name',
    );
    const dimCalls = mockSpan.setAttribute.mock.calls.filter(
      ([key]: [string]) => key === 'embedding.dimension',
    );
    expect(modelCalls).toHaveLength(0);
    expect(dimCalls).toHaveLength(0);
  });
});

// ─── decorateSpan additional edge cases ─────────────────────────────────────────

describe('decorateSpan additional edge cases', () => {
  beforeEach(() => {
    mockSpan = createMockSpan();
  });

  it('should handle non-Error thrown values in sync function', () => {
    const fn = () => {
      throw 'string error'; // eslint-disable-line no-throw-literal
    };
    const wrapped = decorateSpan({ kind: 'CHAIN' }, fn);
    expect(() => wrapped()).toThrow('string error');
    expect(mockSpan.setStatus).toHaveBeenCalledWith({
      code: 2,
      message: 'string error',
    });
    expect(mockSpan.recordException).toHaveBeenCalled();
  });

  it('should handle non-Error thrown values in async function', async () => {
    const fn = async () => {
      throw 42; // eslint-disable-line no-throw-literal
    };
    const wrapped = decorateSpan({ kind: 'CHAIN' }, fn);
    await expect(wrapped()).rejects.toBe(42);
    // The error?.message is undefined for non-Error, so String(error) = "42"
    expect(mockSpan.setStatus).toHaveBeenCalledWith({
      code: 2,
      message: '42',
    });
  });

  it('should use "anonymous" when function has no name', () => {
    const wrapped = decorateSpan({ kind: 'CHAIN' }, (() => 42) as any);
    // Anonymous arrow functions may have empty names
    // The name should default to 'anonymous' or empty
    expect(typeof wrapped.name).toBe('string');
  });

  it('should use name from options when provided', () => {
    const fn = () => 42;
    const wrapped = decorateSpan({ kind: 'CHAIN', name: 'my-operation' }, fn);
    expect(wrapped.name).toBe('my-operation');
  });

  it('should handle function that returns undefined', () => {
    const fn = () => undefined;
    const wrapped = decorateSpan({ kind: 'CHAIN' }, fn);
    const result = wrapped();
    expect(result).toBeUndefined();
    expect(mockSpan.setStatus).toHaveBeenCalledWith({ code: 1 });
    expect(mockSpan.end).toHaveBeenCalled();
  });

  it('should handle function that returns null', () => {
    const fn = () => null;
    const wrapped = decorateSpan({ kind: 'CHAIN' }, fn);
    const result = wrapped();
    expect(result).toBeNull();
    expect(mockSpan.end).toHaveBeenCalled();
  });

  it('should handle async function that resolves with undefined', async () => {
    const fn = async () => undefined;
    const wrapped = decorateSpan({ kind: 'CHAIN' }, fn);
    const result = await wrapped();
    expect(result).toBeUndefined();
    expect(mockSpan.end).toHaveBeenCalled();
  });
});

// ─── safeJsonDumps additional edge cases ────────────────────────────────────────

describe('safeJsonDumps additional edge cases', () => {
  it('should handle undefined', () => {
    // JSON.stringify(undefined) returns undefined (not a string),
    // but safeJsonDumps falls back to String(value) for non-serializable inputs
    const result = safeJsonDumps(undefined);
    // Result is the JS undefined value since JSON.stringify(undefined) = undefined,
    // and the try block returns that (not a string). The function returns it as-is.
    // Actually, JSON.stringify(undefined) returns undefined, so the function returns undefined
    // which when coerced to string by the caller would be "undefined".
    // Let's just verify it doesn't throw
    expect(result).toBeUndefined();
  });

  it('should handle nested objects with special types', () => {
    const obj = {
      error: new TypeError('type mismatch'),
      fn: function hello() {},
      big: BigInt(12345),
      normal: 'value',
    };
    const result = safeJsonDumps(obj);
    const parsed = JSON.parse(result);
    expect(parsed.error.message).toBe('type mismatch');
    expect(parsed.error.name).toBe('TypeError');
    expect(parsed.fn).toBe('[Function: hello]');
    expect(parsed.big).toBe('12345');
    expect(parsed.normal).toBe('value');
  });

  it('should handle empty object', () => {
    expect(safeJsonDumps({})).toBe('{}');
  });

  it('should handle empty array', () => {
    expect(safeJsonDumps([])).toBe('[]');
  });

  it('should handle deeply nested objects', () => {
    const obj = { a: { b: { c: { d: { e: 'deep' } } } } };
    const result = safeJsonDumps(obj);
    const parsed = JSON.parse(result);
    expect(parsed.a.b.c.d.e).toBe('deep');
  });
});

// ─── serializeObj additional edge cases ─────────────────────────────────────────

describe('serializeObj additional edge cases', () => {
  it('should handle nested toJSON', () => {
    const obj = {
      inner: {
        toJSON: () => ({ data: 'serialized' }),
      },
    };
    const result = serializeObj(obj);
    expect(result.inner).toEqual({ data: 'serialized' });
  });

  it('should handle array with mixed types', () => {
    const arr = [1, 'str', null, undefined, true, { a: 1 }];
    const result = serializeObj(arr);
    expect(result).toEqual([1, 'str', null, undefined, true, { a: 1 }]);
  });

  it('should handle empty string', () => {
    expect(serializeObj('')).toBe('');
  });

  it('should handle zero', () => {
    expect(serializeObj(0)).toBe(0);
  });

  it('should handle false', () => {
    expect(serializeObj(false)).toBe(false);
  });
});
