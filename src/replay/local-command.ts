import { spawn } from 'node:child_process';
import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { z } from 'zod';
import { parseDocument } from 'yaml';
import { json } from '../dataset/api.js';
import { parseReplayRecording, replayStatusSchema, type ReplayOptions, type ReplayStatus } from './contracts.js';
import { ReplaySession } from './session.js';

export const agentConfigSchema = z.object({
  command: z.array(z.string().min(1).max(4096)).min(1).max(100),
  input_from_trace: z.record(z.string().regex(/^[A-Za-z][A-Za-z0-9_]*$/), z.string().min(1).max(200)).optional(),
  timeout_seconds: z.number().int().min(1).max(3600).default(300),
}).strict();
export function parseAgentConfig(text: string) {
  if (Buffer.byteLength(text) > 64 * 1024) throw new Error('Agent config exceeds 64 KiB.');
  const document = parseDocument(text, { uniqueKeys: true });
  if (document.errors.length) throw new Error('Agent config is not valid YAML.');
  return agentConfigSchema.parse(document.toJS({ maxAliasCount: 0 }));
}
function mappedInput(value: unknown, mapping?: Record<string, string>): unknown {
  if (!mapping) return value;
  return Object.fromEntries(Object.entries(mapping).map(([key, path]) => {
    let current = value;
    for (const part of path.split('.')) {
      if (!/^[A-Za-z0-9_-]+$/.test(part) || ['__proto__', 'prototype', 'constructor'].includes(part) ||
          current === null || typeof current !== 'object' || !Object.hasOwn(current, part))
        throw new Error(`Agent input mapping does not resolve: ${key}.`);
      current = Reflect.get(current, part);
    }
    return [key, current];
  }));
}
export async function runLocalCommand(options: {
  recording: unknown; replay: ReplayOptions; input: unknown; inputContext?: unknown; config: ReturnType<typeof parseAgentConfig>;
  cwd: string; env?: NodeJS.ProcessEnv; label?: string; params?: Record<string, string | number | boolean | null>;
}): Promise<{ exitCode: number; status: ReplayStatus; stdout: string; stderr: string }> {
  const recording = parseReplayRecording(options.recording);
  new ReplaySession(recording, options.replay);
  const input = JSON.stringify(json(mappedInput(options.config.input_from_trace ? options.inputContext ?? options.input : options.input, options.config.input_from_trace)));
  if (Buffer.byteLength(input) > 16 * 1024 * 1024) throw new Error('Agent input exceeds 16 MiB.');
  const directory = await mkdtemp(join(tmpdir(), 'neatlogs-replay-'));
  try {
    const file = join(directory, 'recording.json'), report = join(directory, 'report.json');
    await writeFile(file, JSON.stringify(recording), { mode: 0o600 });
    const result = await new Promise<{ exitCode: number; stdout: string; stderr: string }>((accept, reject) => {
      const child = spawn(options.config.command[0]!, options.config.command.slice(1), {
        cwd: resolve(options.cwd), shell: false, detached: process.platform !== 'win32',
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...(options.env ?? process.env), NEATLOGS_REPLAY_FILE: file, NEATLOGS_REPLAY_REPORT: report,
          NEATLOGS_REPLAY_TRACE_ID: recording.sourceTraceId, NEATLOGS_REPLAY_MODE: options.replay.mode,
          NEATLOGS_REPLAY_UNTIL: options.replay.untilSpan ?? '', NEATLOGS_REPLAY_LABEL: options.label ?? 'local-replay',
          NEATLOGS_REPLAY_RUN_ID: options.replay.run?.runId ?? '', NEATLOGS_REPLAY_CASE_ID: options.replay.run?.caseId ?? '',
          NEATLOGS_REPLAY_TRIAL: options.replay.run ? String(options.replay.run.trial) : '',
          NEATLOGS_REPLAY_PARAMS: JSON.stringify(options.params ?? {}) },
      });
      let stdout = '', stderr = '', bytes = 0, failure: Error | undefined;
      const stop = (error: Error) => {
        failure ??= error;
        try { if (child.pid && process.platform !== 'win32') process.kill(-child.pid, 'SIGKILL'); else child.kill('SIGKILL'); }
        catch { /* The process can exit between the limit check and cancellation. */ }
      };
      const timer = setTimeout(() => stop(new Error('Local agent exceeded its time limit.')), options.config.timeout_seconds * 1000);
      const collect = (kind: 'stdout' | 'stderr', chunk: Buffer) => {
        bytes += chunk.length;
        if (bytes > 16 * 1024 * 1024) { stop(new Error('Local agent output exceeds 16 MiB.')); return; }
        if (kind === 'stdout') stdout += chunk.toString('utf8'); else stderr += chunk.toString('utf8');
      };
      child.stdout.on('data', chunk => collect('stdout', chunk));
      child.stderr.on('data', chunk => collect('stderr', chunk));
      child.on('error', error => { clearTimeout(timer); reject(error); });
      child.stdin.on('error', error => { if ('code' in error && error.code !== 'EPIPE') stop(error); });
      child.on('close', code => {
        clearTimeout(timer);
        if (failure) reject(failure); else accept({ exitCode: code ?? 1, stdout, stderr });
      });
      child.stdin.end(`${input}\n`);
    });
    let status: ReplayStatus;
    try {
      if ((await stat(report)).size > 16 * 1024) throw new Error('Replay activity report exceeds 16 KiB.');
      status = replayStatusSchema.parse(JSON.parse(await readFile(report, 'utf8')));
    }
    catch { throw new Error('The agent did not report replay activity. Call init() with a replay-capable SDK in the child process.'); }
    if (status.sourceTraceId !== recording.sourceTraceId || status.mode !== options.replay.mode)
      throw new Error('Local replay report does not match the requested recording.');
    if (status.hits + status.liveCalls === 0) throw new Error('No replay-aware calls ran. Mark JSON tools with replay: true or wrap model calls with replayModel().');
    return { ...result, status };
  } finally { await rm(directory, { recursive: true, force: true }); }
}
