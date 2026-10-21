import { readFile, stat } from 'node:fs/promises';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { DatasetApi, record, uuid } from './dataset/api.js';
import { parseReplayRecording, replaySettingsSchema } from './replay/contracts.js';
import { replayRecordingFromCase } from './replay/frozen-case.js';
import { parseAgentConfig, runLocalCommand } from './replay/local-command.js';
import { LocalRunApi } from './replay/run-api.js';
import { ReplaySession } from './replay/session.js';
import { continueWholeDataset, startWholeDataset } from './replay/all-command.js';
import { agentConfigDigest } from './replay/all-state.js';

const usage = `Usage: neatlogs simulate local --recording <file> [--until-span <id>] [--json]
   or: neatlogs simulate local --dataset <id> --version <id> --case <id> [--case <id> ...]
   or: neatlogs simulate local --dataset <id> --version <id> --all --state <file> [--batch-size 25]
   or: neatlogs simulate local --resume <file> [--json]
Add --upload --form <id> --evaluator <id> to save a graded run. Repeat --evaluator for more graders.
--trials 1..10, --agent <path>, --label <name>, --param name=value and --mode tools_only|until_span are optional.
The local command comes from .neatlogs/agent.yaml. Tools use saved results; models stay live.
On the first missing match, subsequent calls run live, including tools.
Dataset/API access uses NEATLOGS_TOKEN and NEATLOGS_PROJECT_ID. The agent must separately export its trace and call shutdown().`;
async function boundedFile(path: string, maxBytes: number): Promise<string> {
  if ((await stat(path)).size > maxBytes) throw new Error('Local replay file exceeds the size limit.');
  return readFile(path, 'utf8');
}
function numberOption(raw: string, field: string, min: number, max: number): number {
  const value = Number(raw);
  if (!Number.isInteger(value) || value < min || value > max) throw new Error(`${field} must be from ${min} to ${max}.`);
  return value;
}
function parameters(raw: string[]): Record<string, string | number | boolean | null> {
  if (raw.length > 20) throw new Error('Use at most 20 parameters.');
  const pairs = raw.map(param => {
    const separator = param.indexOf('=');
    if (separator < 1 || separator > 100 || param.length > 2101) throw new Error('Use --param name=value, up to 2000 characters per value.');
    const name = param.slice(0, separator), text = param.slice(separator + 1);
    let value: unknown = text;
    try { value = JSON.parse(text); } catch { /* Unquoted text remains a string. */ }
    if (value !== null && typeof value !== 'string' && typeof value !== 'boolean' && !(typeof value === 'number' && Number.isFinite(value)))
      throw new Error('Parameters must be strings, finite numbers, booleans or null.');
    return [name, value] as const;
  });
  if (new Set(pairs.map(([name]) => name)).size !== pairs.length) throw new Error('Specify each parameter once.');
  return Object.fromEntries(pairs);
}
const apiOptions = () => ({ token: process.env.NEATLOGS_TOKEN ?? '', projectId: process.env.NEATLOGS_PROJECT_ID ?? '', baseUrl: process.env.NEATLOGS_HOST ?? 'https://app.neatlogs.com' });
type CommandResult = Awaited<ReturnType<typeof runLocalCommand>>;
type Report = { caseId: string | null; trial: number; skippedTools: number; uploaded: boolean; result?: CommandResult; error?: string; outputId?: string };
export async function runSimulateCli(argv: string[]): Promise<number> {
  if (argv.includes('--help') || argv.includes('-h')) { process.stdout.write(`${usage}\n`); return 0; }
  let activeRunId: string | undefined;
  try {
    if (argv[0] !== 'simulate' || argv[1] !== 'local') throw new Error(usage);
    const { values } = parseArgs({ args: argv.slice(2), strict: true, options: {
      recording: { type: 'string' }, dataset: { type: 'string' }, version: { type: 'string' }, case: { type: 'string', multiple: true },
      agent: { type: 'string', default: '.neatlogs/agent.yaml' }, mode: { type: 'string' }, 'until-span': { type: 'string' },
      label: { type: 'string', default: 'Local replay' }, param: { type: 'string', multiple: true }, json: { type: 'boolean', default: false },
      upload: { type: 'boolean', default: false }, form: { type: 'string' }, evaluator: { type: 'string', multiple: true },
      trials: { type: 'string', default: '1' }, 'deadline-seconds': { type: 'string', default: '900' },
      all: { type: 'boolean', default: false }, state: { type: 'string' }, resume: { type: 'string' }, 'batch-size': { type: 'string' },
    } });
    if (values.resume) {
      if (argv.slice(2).some(value => value.startsWith('--') && !['--resume', '--json'].includes(value.split('=')[0]!)))
        throw new Error('Continue with --resume <state-file> and optional --json only. Saved settings remain unchanged.');
      return await continueWholeDataset({ path: values.resume, apiOptions: apiOptions(), json: values.json! });
    }
    if (values.all && (!values.dataset || values.recording || values.case?.length || !values.state))
      throw new Error('Use --all with --dataset, --version and a new --state file; omit --case and --recording.');
    if (!values.all && (values.state || values['batch-size'])) throw new Error('--state and --batch-size require --all.');
    if (Boolean(values.recording) === Boolean(values.dataset)) throw new Error('Choose either a recording file or pinned dataset examples.');
    if (values.recording && (values.version || values.case?.length)) throw new Error('Dataset version/case flags cannot be used with a recording file.');
    if (values.upload && !values.dataset) throw new Error('Upload requires a pinned dataset version and examples.');
    if (!values.upload && (values.form || values.evaluator?.length)) throw new Error('Use --upload to apply grading settings.');
    const mode = values.mode ?? (values['until-span'] ? 'until_span' : 'tools_only');
    if (mode !== 'tools_only' && mode !== 'until_span' || mode === 'until_span' && !values['until-span'] || mode === 'tools_only' && values['until-span'])
      throw new Error('Use tools_only, or supply --until-span for checkpoint replay.');
    const replay = replaySettingsSchema.parse({ mode, ...(values['until-span'] ? { untilSpan: values['until-span'] } : {}) });
    const trials = numberOption(values.trials!, 'trials', 1, 10), deadlineSeconds = numberOption(values['deadline-seconds']!, 'deadline-seconds', 60, 3600);
    const params = parameters(values.param ?? []), config = parseAgentConfig(await boundedFile(resolve(values.agent!), 64 * 1024));
    const label = values.label!.trim();
    if (!label || label.length > 120) throw new Error('Use a label from 1 to 120 characters.');
    const dataset = values.dataset ? uuid(values.dataset, 'dataset') : undefined;
    const version = dataset ? uuid(values.version, 'version') : undefined;
    const caseIds = (values.case ?? []).map(value => uuid(value, 'case'));
    if (dataset && !values.all && (!caseIds.length || caseIds.length > 100 || new Set(caseIds).size !== caseIds.length)) throw new Error('Select 1 to 100 different examples with --case.');
    const grading = values.upload ? { formVersionId: uuid(values.form, 'form'), evaluatorIds: (values.evaluator ?? []).map(value => uuid(value, 'evaluator')) } : undefined;
    if (grading && (!grading.evaluatorIds.length || grading.evaluatorIds.length > 20 || new Set(grading.evaluatorIds).size !== grading.evaluatorIds.length)) throw new Error('Choose 1 to 20 different evaluators.');
    if (values.all && dataset && version && values.state) {
      const api = apiOptions();
      return await startWholeDataset({ path: values.state, apiOptions: api, agent: config, json: values.json!, configuration: {
        projectId: uuid(api.projectId, 'projectId'), baseUrl: new URL(api.baseUrl).origin, datasetId: dataset, versionId: version,
        cwd: resolve(process.cwd()), agentFile: resolve(values.agent!), agentDigest: agentConfigDigest(config), label, trials, params, replay, deadlineSeconds,
        grading: grading ?? null, batchSize: numberOption(values['batch-size'] ?? '25', 'batch-size', 1, 100),
      } });
    }
    const datasetApi = dataset ? new DatasetApi(apiOptions()) : undefined;
    const sources: { caseId: string | null; item: Record<string, unknown>; converted: ReturnType<typeof replayRecordingFromCase> }[] = [];
    let bytes = 0;
    for (const caseId of dataset ? caseIds : [null]) {
      const source = datasetApi && caseId ? record(await datasetApi.request('GET', `/datasets/${dataset}/cases/${caseId}?versionId=${version}`)).case
        : JSON.parse(await boundedFile(resolve(values.recording!), 16 * 1024 * 1024));
      const item = record(source);
      if (caseId && item.caseId !== caseId) throw new Error('Dataset API returned a different example.');
      bytes += Buffer.byteLength(JSON.stringify(item));
      if (bytes > 32 * 1024 * 1024) throw new Error('Selected recordings exceed 32 MiB. Select fewer examples per run.');
      const converted = 'entries' in item ? { recording: parseReplayRecording(item), skippedTools: 0 } : replayRecordingFromCase(item);
      new ReplaySession(converted.recording, { mode, ...(values['until-span'] ? { untilSpan: values['until-span'] } : {}) });
      sources.push({ caseId, item, converted });
    }
    const api = grading ? new LocalRunApi(apiOptions()) : undefined;
    const run = api && grading && version ? await api.register({ datasetVersionId: version, label, caseIds, trials, params, grading, deadlineSeconds, replay }) : undefined;
    activeRunId = run?.runId;
    if (run) process.stderr.write(`Registered run ${run.runId}; trace upload will wait for delivery.\n`);
    const reports: Report[] = [];
    let outputBytes = 0, outputLimitReached = false;
    for (const { caseId, item, converted } of sources) for (let trial = 1; trial <= trials; trial++) {
      const report: Report = { caseId, trial, skippedTools: converted.skippedTools, uploaded: false };
      reports.push(report);
      try {
        if (outputLimitReached) throw new Error('The command output limit was reached; this example was not executed.');
        const remaining = run ? Date.parse(run.deadlineAt) - Date.now() : Infinity;
        if (remaining < 1000) throw new Error('The run deadline has passed; this example was not executed.');
        const result = await runLocalCommand({ recording: converted.recording, input: 'entries' in item ? null : item.input,
          inputContext: { ...item, root: { input: item.input ?? null, output: item.observedOutput ?? null } },
          replay: { mode, ...(values['until-span'] ? { untilSpan: values['until-span'] } : {}), ...(run && caseId ? { run: { runId: run.runId, caseId, trial } } : {}) },
          config: { ...config, timeout_seconds: Math.min(config.timeout_seconds, Math.max(1, Math.floor(remaining / 1000))) }, cwd: process.cwd(), label, params });
        outputBytes += Buffer.byteLength(result.stdout) + Buffer.byteLength(result.stderr);
        if (outputBytes > 32 * 1024 * 1024) {
          outputLimitReached = true;
          throw new Error('Combined command output exceeds 32 MiB. Run fewer examples or reduce agent logging.');
        }
        report.result = result;
        if (result.exitCode !== 0) throw new Error(`Agent exited with code ${result.exitCode}; no output was uploaded.`);
        if (api && run && caseId) {
          if (result.status.traceIds.length !== 1) throw new Error('Upload requires exactly one workflow trace. Wrap the agent in one root span and call shutdown().');
          const receipt = await api.upload(run.runId, { caseId, trial, traceId: result.status.traceIds[0]! }, { waitMillis: Math.max(1, Math.min(60_000, Date.parse(run.deadlineAt) - Date.now())) });
          report.uploaded = true; report.outputId = receipt.outputId;
        }
      } catch (error) { report.error = error instanceof Error ? error.message : 'Local replay failed.'; }
      if (!values.json) {
        if (report.result) { process.stdout.write(report.result.stdout); process.stderr.write(report.result.stderr); }
        process.stdout.write(`\n${caseId ?? 'Recording'}, attempt ${trial}: ${report.error ?? (report.uploaded ? 'Saved to Runs; grading is pending.' : 'Finished locally; not uploaded.')}\n`);
      }
    }
    const exitCode = reports.some(report => report.error) ? 1 : 0;
    if (values.json) {
      const first = reports[0]!;
      const output = reports.length === 1 ? { ...first.result, ...first, result: undefined, exitCode, runId: run?.runId } : { exitCode, runId: run?.runId, reports };
      process.stdout.write(`${JSON.stringify(output)}\n`);
    }
    return exitCode;
  } catch (error) { process.stderr.write(`${error instanceof Error ? error.message : 'Local replay failed.'}${activeRunId ? ` Run: ${activeRunId}.` : ''}\n`); return 1; }
}
