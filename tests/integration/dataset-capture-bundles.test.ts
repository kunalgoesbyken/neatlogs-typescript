import { execFileSync } from 'node:child_process';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

describe('dataset capture across published package entrypoints', () => {
  it('records a Google tool call ID on the exported tool span', () => {
    const corePath = resolve('dist/index.mjs');
    const integrationPath = resolve('dist/google-genai.mjs');
    const script = `
      import { pathToFileURL } from 'node:url';
      const { Client, trace } = await import(pathToFileURL(${JSON.stringify(corePath)}).href);
      const { traceTool } = await import(pathToFileURL(${JSON.stringify(integrationPath)}).href);
      const exported = [];
      const exporter = { export(spans, done) { exported.push(...spans); done({ code: 0 }); }, async shutdown() {}, async forceFlush() {} };
      const client = new Client({ apiKey: 'unused', workflowName: 'tool-call-id', spanExporter: exporter });
      try {
        const tool = traceTool('lookup', async () => 'result', { callId: 'provider-call-123' });
        await client.activate(() => trace({ name: 'agent', kind: 'WORKFLOW' }, () => tool({})));
        await client.flush();
        console.log(JSON.stringify(exported.find((span) => span.name === 'tool.lookup')?.attributes));
      } finally { await client.shutdown(); }
    `;
    const attributes = JSON.parse(execFileSync(process.execPath, ['--input-type=module', '-e', script], { encoding: 'utf8', timeout: 15_000 }));
    expect(attributes['neatlogs.tool.call_id']).toBe('provider-call-123');
  }, 20_000);
  it.each([['mjs', 'mjs'], ['cjs', 'cjs'], ['mjs', 'cjs']])('shares the capture scope between index.%s and openai.%s', (coreFormat, integrationFormat) => {
    const corePath = resolve(`dist/index.${coreFormat}`);
    const integrationPath = resolve(`dist/openai.${integrationFormat}`);
    const script = `
      import { createRequire } from 'node:module';
      import { pathToFileURL } from 'node:url';
      const require = createRequire(import.meta.url);
      const load = async (path) => path.endsWith('.cjs') ? require(path) : import(pathToFileURL(path).href);
      const { Client, datasets, trace } = await load(${JSON.stringify(corePath)});
      const { traceTool } = await load(${JSON.stringify(integrationPath)});
      const id = '11111111-1111-4111-8111-111111111111';
      const datasetId = '22222222-2222-4222-8222-222222222222';
      const exported = [];
      let traces;
      globalThis.fetch = async (url, init) => {
        if (url.endsWith('/complete')) traces = JSON.parse(init.body).traces;
        return Response.json({ success: true, data: { captureId: id, datasetId, state: url.endsWith('/complete') ? 'queued' : 'recording' } });
      };
      const exporter = { export(spans, done) { exported.push(...spans); done({ code: 0 }); }, async shutdown() {}, async forceFlush() {} };
      const client = new Client({ apiKey: 'unused', workflowName: 'bundle-capture', spanExporter: exporter });
      try {
        const tool = traceTool('real-tool', async () => { await Promise.resolve(); return 'actual-tool-result'; });
        const capture = await client.activate(() => datasets.capture({ kind: 'existing', datasetId }, () => trace({ name: 'agent', kind: 'WORKFLOW' }, () => tool({ question: 'actual' })), {
          token: 'dataset-token', projectId: '33333333-3333-4333-8333-333333333333', baseUrl: 'https://api.example.test', environment: 'ci'
        }));
        const spans = exported.filter((span) => span.name !== 'neatlogs.trace.complete');
        console.log(JSON.stringify({ traces, spanIds: spans.map((span) => span.spanContext().spanId), names: spans.map((span) => span.name), result: capture.result }));
      } finally { await client.shutdown(); }
    `;
    const result = JSON.parse(execFileSync(process.execPath, ['--input-type=module', '-e', script], { encoding: 'utf8', timeout: 15_000 }));
    expect(result.result).toBe('actual-tool-result');
    expect(result.names.sort()).toEqual(['agent', 'tool.real-tool']);
    expect(result.traces).toHaveLength(1);
    expect(result.traces[0].spanIds.sort()).toEqual(result.spanIds.sort());
  }, 20_000);
});

