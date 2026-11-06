# Local replay

Run your own agent code while reusing saved tool results. This is useful when changing orchestration or parsing that a hosted model-call simulation cannot run.

The SDK supports selected examples or an entire frozen dataset, repeated attempts, and uploading completed traces for grading. These commands require a package version that includes local replay. Python also supports local replay; Go does not yet provide a local replay runner.

## Run a saved example

Mark JSON-returning tools that may use a recording:

```ts
import { init, span, shutdown } from 'neatlogs';

await init();
const lookupOrder = span(
  { kind: 'TOOL', toolName: 'lookup_order', replay: true },
  async (input: { orderId: string }) => fetchOrder(input.orderId),
);
// Read one JSON input from stdin and run your normal agent.
// Finish by flushing telemetry through shutdown().
await shutdown();
```

Commit `.neatlogs/agent.yaml` with the command as an argument list. The runner does not invoke a shell:

```yaml
command: [node, dist/replay-entry.mjs]
timeout_seconds: 300
input_from_trace:
  message: root.input.message
  customer_id: metadata.customer_id
```

Without `input_from_trace`, the agent receives the saved example's input unchanged. The command runs from the current directory and receives one JSON value on stdin. `--param model=small` is available as `JSON.parse(process.env.NEATLOGS_REPLAY_PARAMS)`; the entry point applies it to the agent.

```sh
neatlogs simulate local --dataset DATASET_UUID --version VERSION_UUID --case CASE_UUID --param model=small --json
```

Dataset reads use `NEATLOGS_TOKEN`, `NEATLOGS_PROJECT_ID`, and optionally `NEATLOGS_HOST`. The token needs `evaluation:read`. A frozen example saved as JSON can be used without API access:

```sh
neatlogs simulate local --recording example.json --json
```

## What runs live

`tools_only` is the default. Model calls run live. An opted-in tool returns its saved JSON value when its name and arguments match exactly, ignoring JSON object key order. Repeated identical calls consume saved results in recorded order. A cached value is copied so changing it cannot change the next result.

**The first missing match switches the remaining execution to live code, including tools.** Use a local development environment whose real tool actions are appropriate to run. The result reports the first miss and saved/live call counts. This is different from hosted simulation, which never executes real tools.

Sync tools stay synchronous. Native async functions return promises. For a function that returns a promise without being declared `async`, use `replay: { result: 'promise' }`. Cached tools must return JSON values; streams, class instances, and functions need a custom adapter.

Every generated SDK span is marked with simulation origin. Uploaded runs also bind each span to its run, example and attempt. The runner bounds input, output and runtime, removes temporary recordings after the command ends, and fails if the child SDK did not report replay activity.

## Checkpoint replay

The `ReplaySession` API accepts a versioned recording with entries `{spanId, kind, name, input, output}`. `kind` is `tool` or `model`. A checkpoint is the **first call executed live**; later entries cannot be served before that checkpoint.

Wrapped OpenAI Chat Completions, OpenAI Responses and Google `models.generateContent` calls automatically look up model entries. Their entry names are `openai.chat.completions.create`, `openai.responses.create` and `google_genai.models.generate_content`. The entry input is the exact JSON request body; output must be the complete provider response. OpenAI Responses entries include `output_text`. Google restores the installed SDK response class, including its text and tool-call getters. Cached telemetry records zero newly billed tokens while application code receives the original recorded usage.

The wrapped OpenAI Chat Completions, OpenAI Responses and Google `models.generateContent` integrations can save bounded raw request/response captures in protected trace payloads. Frozen datasets can restore model responses only from complete, verified captures. Masked, redacted, truncated or otherwise incomplete evidence is excluded; normalized answer text is never treated as a complete provider object. Missing or unsupported evidence, streaming calls, extra transport options and mutable Google chat sessions cause a cache miss and switch the remaining run to live code. `tools_only` keeps model calls live without disabling tool hits. A manual HTTP adapter does not gain model-prefix replay from these SDK hooks; the Flipkart example uses that adapter with `tools_only`.

Other model adapters can use `replayModel(name, input, decodeSavedResponse, liveCall)`. The decoder validates saved JSON and reconstructs the return type expected by your code.

```ts
const session = new ReplaySession(recording, {
  mode: 'until_span', untilSpan: checkpointId,
});
const result = await session.run(() => runAgent(input));
console.log(session.getStatus());
```

OpenAI HTTP response helpers (`withResponse` and `asResponse`) must be selected before consuming the response. They execute a recorded live miss because saved model data does not include faithful HTTP metadata. Calling a raw helper after consuming its cached response, or after consuming a later reserved cached response, rejects explicitly without making an additional request. Duplicate requests reserve recorded responses in invocation order, independently of await order.

The CLI accepts `--until-span ID` for a compatible recording. Replay sessions are isolated between concurrent callbacks and can be used once. Nested sessions are rejected.

## Save results in Runs

Use a public API token with `evaluation:read` and `evaluation:write`. Your agent's ingestion API key is configured separately through normal SDK initialization. Wrap the entire agent in one root workflow trace, then call `shutdown()` before the process exits.

```sh
neatlogs simulate local \
  --dataset DATASET_UUID --version VERSION_UUID \
  --case CASE_UUID --case ANOTHER_CASE_UUID --trials 2 \
  --upload --form FORM_VERSION_UUID --evaluator EVALUATOR_UUID \
  --param model=small --param temperature=0.2 --label 'Smaller model'
```

Each selected example runs in a fresh child process. The server freezes the grading setup before execution and compares the complete saved example with the accepted generated trace. Only one trace can fill each example/attempt slot; multiple traces are rejected instead of choosing one silently. Failed child processes are not uploaded. The API waits briefly for trace delivery using one stable idempotency key and retries only `TRACE_NOT_READY`. An uploaded result means the trace was accepted; grading can still be pending.

New uploaded runs also save the exact replay mode and checkpoint. **Run again** restores those settings in the command. Older runs without saved settings display a warning and default to `tools_only`; their original checkpoint cannot be recovered.

`--param` values accept JSON primitives (number, boolean, null, or a quoted JSON string); other text stays a string. Each parameter name is unique. Repeat `--evaluator` for additional graders. Runs allow up to 100 selected examples, 10 attempts per example, 32 MiB of loaded recordings and 32 MiB of combined process output. The default acceptance deadline is 900 seconds; use `--deadline-seconds 60..3600` when needed. Deadline expiry or stopping acceptance in the app does not remotely terminate a process on your computer.

Without `--upload`, the same command keeps results local. A local recording file can never be submitted as a substitute for a pinned server dataset.

## Entire saved dataset

Use `--all` instead of repeating `--case`. Whole-dataset replay supports `tools_only`; `--all` with `--until-span` is rejected before registration because each recording has its own checkpoint IDs. A version UUID is required, so later edits to the dataset cannot change this selection.

```sh
neatlogs simulate local \
  --dataset DATASET_UUID --version VERSION_UUID --all \
  --state .neatlogs/replays/my-run.json --batch-size 25 \
  --upload --form FORM_VERSION_UUID --evaluator EVALUATOR_UUID \
  --param model=small --label 'Smaller model'
```

The CLI reads all example IDs and revision IDs in bounded pages before starting your agent. It rejects incomplete pages, repeated IDs and changed totals. The current frozen-version pagination limit is 2,000 examples; a larger version is rejected before execution. This does not raise the existing 100-example limit per saved run. Instead, the CLI creates separate runs with `[1/N]` added to their labels, loads one recording at a time, and prints every run ID. `--batch-size` accepts 1–100 and defaults to 25. Each batch gets its own acceptance deadline when registered. Each child process retains the existing input, output and time limits.

`--state` must name a new file. It saves the exact version, ordered selection digest, parameters, grading, replay settings, batch IDs, stable retry keys, replay activity reports and accepted output receipts. It does not store API credentials, raw recordings or agent output. Agent command settings are stored as a digest; the original agent YAML remains required. The file is private to the current OS user and limited to 16 MiB. If another activity report would exceed that bound, the next command is not started.

`--json` returns `reports` for each executed attempt, including its source trace, run ID when uploaded, exit code and the SDK's saved/live call counts, first miss and resulting trace IDs. These reports survive upload retries and resume, including local-only completions and failed commands that returned a valid report. Normal terminal output shows HIT/MISS/LIVE counts for each attempt. An interrupted command with an unknown outcome has no invented report.

Continue after an interrupted API request with the same project, API origin, working directory and agent YAML:

```sh
neatlogs simulate local --resume .neatlogs/replays/my-run.json
```

Only `--resume` and optional `--json` are accepted when continuing. The CLI reuses registration keys and retries an already produced trace upload without running that command again. Accepted attempts are skipped. Closed or expired batches cannot start more commands. Failed commands are not automatically repeated. If a process died while a command was running, its outcome is unknown and continuation stops for inspection. An abrupt process exit can leave a `.lock` file; remove it only after checking that no process still owns the saved run. Agent YAML is checked on resume, but the CLI does not snapshot your application's source code.

`--all` cannot be combined with `--case` or `--recording`. Without `--upload`, the same selection runs locally and keeps progress in the state file. In the app, a saved batch's **Run again** command selects that batch's exact examples; it does not silently repeat the entire dataset.

## Task evaluation comparison pages

Graded runs require every selected case to complete; `maxErrorRate` must stay at its default zero. A failed attempt produces an incomplete comparison, never a relaxed quality pass.

`runEval` returns the first detail page in `result.comparison`. Its coverage, question summaries and server-side gates describe the whole run. If `nextCursor` is present, use `LocalRunApi.comparison(runId, { after: nextCursor, limit: 100, baseline })` to read the next page. Preserve the same baseline on every page (`saved` or `run:<UUID>`). The method also supports `newFailuresOnly`; the server validates baseline compatibility and cursor membership.
