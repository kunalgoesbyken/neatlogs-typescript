import { readFileSync, statSync, writeFileSync } from 'node:fs';
import { ReplaySession, setEnvironmentReplay } from './session.js';
import { replayRunBindingSchema } from './contracts.js';
const marker = Symbol.for('neatlogs.local_replay_environment_loaded');
type Global = typeof globalThis & { [marker]?: boolean };

/** Only the explicit local CLI sets this environment; never fetches on the request path. */
export function configureReplayFromEnvironment(): void {
  const file = process.env.NEATLOGS_REPLAY_FILE;
  if (!file || (globalThis as Global)[marker]) return;
  if (statSync(file).size > 16 * 1024 * 1024) throw new Error('Replay recording exceeds 16 MiB.');
  const mode = process.env.NEATLOGS_REPLAY_MODE ?? 'tools_only';
  if (mode !== 'tools_only' && mode !== 'until_span') throw new Error('Unsupported local replay mode.');
  const run = process.env.NEATLOGS_REPLAY_RUN_ID ? replayRunBindingSchema.parse({
    runId: process.env.NEATLOGS_REPLAY_RUN_ID, caseId: process.env.NEATLOGS_REPLAY_CASE_ID,
    trial: Number(process.env.NEATLOGS_REPLAY_TRIAL),
  }) : undefined;
  const session = new ReplaySession(JSON.parse(readFileSync(file, 'utf8')), {
    mode, ...(process.env.NEATLOGS_REPLAY_UNTIL ? { untilSpan: process.env.NEATLOGS_REPLAY_UNTIL } : {}),
    ...(run ? { run } : {}),
  });
  setEnvironmentReplay(session);
  (globalThis as Global)[marker] = true;
  const report = process.env.NEATLOGS_REPLAY_REPORT;
  if (report) process.once('exit', () => {
    try { writeFileSync(report, JSON.stringify(session.getStatus()), { mode: 0o600 }); }
    catch { process.stderr.write('Neatlogs could not save the local replay report.\n'); }
  });
}
