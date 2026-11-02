import { defineConfig } from 'tsup';

export default defineConfig({
  entry: ['src/evals.ts', 'src/index.ts', 'src/cli.ts', 'src/ai-sdk.ts', 'src/openai.ts', 'src/anthropic.ts', 'src/azure-openai.ts', 'src/vertex-ai.ts', 'src/google-genai.ts', 'src/bedrock.ts', 'src/langchain.ts', 'src/strands.ts', 'src/openai-agents.ts', 'src/mastra-wrap.ts', 'src/pi-agent.ts', 'src/claude-agent-sdk.ts', 'src/openrouter-agent.ts', 'src/opencode-plugin.ts', 'src/browser.ts'],
  format: ['cjs', 'esm'],
  dts: true,
  splitting: false,
  sourcemap: true,
  clean: true,
  target: 'node18',
  outExtension({ format }) {
    return {
      js: format === 'cjs' ? '.cjs' : '.mjs',
    };
  },
});
