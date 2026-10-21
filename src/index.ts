/**
 * Neatlogs TypeScript SDK
 *
 * OpenTelemetry-native observability for LLM applications.
 *
 * @example
 * ```typescript
 * import { init, span, shutdown } from 'neatlogs';
 *
 * async function main() {
 *   await init({ apiKey: process.env.NEATLOGS_API_KEY });
 *
 *   const myWorkflow = span({ kind: 'WORKFLOW' }, async (query: string) => {
 *     // your LLM code here
 *   });
 *
 *   await myWorkflow('Hello!');
 *   await shutdown();
 * }
 * main().catch(console.error);
 * ```
 */

// Lifecycle
export {
  init,
  flush,
  flushAll,
  flushAllDetailed,
  shutdown,
  isDebugEnabled,
  getSessionConfig,
  getDeliveryDiagnostics,
} from "./init.js";
export { Client, type ClientOptions } from "./core/client.js";
export { AbTestClient, type AbTestClientOptions, type AbAssignment, type AbAssignmentOptions, type LogExposureOptions } from './ab-test/client.js';
export { abTest, logExposure, completeAbTestUnit } from './ab-test/shared.js';
export type { AbParams } from './ab-test/protocol.js';
export { ReplaySession, getReplayStatus, replayModel } from './replay/session.js';
export { parseReplayRecording } from './replay/contracts.js';
export { replayRecordingFromCase } from './replay/frozen-case.js';
export type { ReplayRecording, ReplayEntry, ReplayOptions, ReplayStatus, ReplayRunBinding } from './replay/contracts.js';
export { LocalRunApi, LocalRunApiError } from './replay/run-api.js';
export { datasets, DatasetCaptureHandle, DatasetCaptureError, getDatasetCaptureHandle } from './dataset/capture.js';
export type { DatasetCaptureOptions, DatasetCaptureResult, DatasetCaptureWaitOptions, DatasetCaptureStatusOptions, DatasetCaptureDestination, DatasetCaptureState, DatasetCaptureStatus, DatasetCaptureFailure } from './dataset/capture.js';
export type { DatasetApiOptions, DatasetJson } from './dataset/api.js';
export type { CreateDatasetInput, DatasetCaseInput, DatasetMutationOptions, DatasetVersion, DatasetOutcomeChange } from './dataset/mutations.js';
export {
  NeatlogsConfigurationError,
  type NeatlogsConfigurationErrorCode,
} from "./errors.js";

// Instrumentation
export { span, Span } from "./decorators/index.js";
export { trace, setTraceOutput } from "./core/context.js";
export { log } from "./core/log.js";

// Request-scoped session & end-user identity (per-request, not process-global).
export { identify, type IdentifyOptions } from "./core/identity.js";
export {
  extractTraceContext,
  injectTraceContext,
  type TraceContextCarrier,
} from "./core/propagation.js";

// Prompt management
export { PromptTemplate, UserPromptTemplate } from "./prompt/template.js";
export {
  PromptClient,
  PromptHandle,
  PromptClientError,
  PromptApiError,
  PromptNotFoundError,
  PromptRequestTimeoutError,
  PromptClientClosedError,
  getPrompt,
  fetchPrompt,
  listPrompts,
  createPrompt,
  updatePrompt,
  saveAsVersion,
  deletePrompt,
  setLabel,
  addTag,
  removeTag,
  type GetPromptOptions,
  type PromptClientOptions,
  type PromptMutationSelector,
  type PromptWriteInput,
  type CreatePromptInput,
  type SavePromptVersionOptions,
  type LegacySavePromptVersionOptions,
} from "./prompt/client.js";

// Mastra integration
export { getMastraObservability } from "./mastra.js";

// Vercel AI SDK integration
export { wrapAISDK, createAITelemetry } from "./ai-sdk.js";

// Provider wrappers
export { wrapOpenAI, traceTool as traceToolOpenAI } from "./openai.js";
export { wrapAnthropic, traceTool as traceToolAnthropic } from "./anthropic.js";
export {
  wrapAzureOpenAI,
  traceTool as traceToolAzureOpenAI,
} from "./azure-openai.js";
export {
  wrapVertexAI,
  wrapVertexAIChat,
  traceTool as traceToolVertexAI,
} from "./vertex-ai.js";
export {
  wrapGoogleGenAI,
  wrapGoogleGenAIChat,
  traceTool as traceToolGoogleGenAI,
} from "./google-genai.js";
export { wrapBedrock, traceTool as traceToolBedrock } from "./bedrock.js";

// Agent framework integrations
export { langchainHandler } from "./langchain.js";
export { strandsHooks } from "./strands.js";
export { openaiAgentsProcessor } from "./openai-agents.js";
export { wrapMastra } from "./mastra-wrap.js";
export { piAgentHooks, tracePiAgentEvents, tracePiStream } from "./pi-agent.js";
export { wrapClaudeAgentSDK } from "./claude-agent-sdk.js";
export { wrapOpenRouterAgent, wrapCallModel } from "./openrouter-agent.js";
export { NeatlogsOpencodePlugin } from "./opencode-plugin.js";

// Utilities
export { bindTemplates } from "./core/llm-binder.js";
export { registerCrewaiTask } from "./core/crewai-task-registry.js";
export {
  TELEMETRY_CONTRACT_VERSION,
  TELEMETRY_SCHEMA_VERSION,
  TELEMETRY_SCHEMA_SHA256,
  TELEMETRY_SCHEMA_V2,
  TELEMETRY_CONFLICT_PRECEDENCE,
} from "./schema-v2.js";
export {
  DOCTOR_V2_FORMAT_VERSION,
  doctorLocalV2,
  doctorCapturedLocalV2,
  doctorSemanticDigest,
} from './doctor-v2.js';
export type {
  DiagnosticEnvelope,
  DiagnosticSpan,
  DoctorV2Check,
  DoctorV2Result,
} from './doctor-v2.js';

// Types
export type {
  InitOptions,
  SpanOptions,
  TraceOptions,
  TraceSpanKind,
  SpanKind,
  MaskFunction,
  MaskContext,
  CachedPrompt,
  PromptMessage,
} from "./types.js";
export type { DeliveryDiagnosticsSnapshot } from "./core/delivery-diagnostics.js";
export type { FlushAllResult, FlushOutcome } from "./init.js";
export {
  HttpUploadAuthority,
  DisabledUploadAuthority,
  TelemetryUploadError,
  type HttpUploadAuthorityOptions,
  type UploadAuthority,
  type UploadAuthorityOption,
  type UploadContentEncoding,
  type UploadPayload,
  type UploadPayloadSchema,
  type UploadPurpose,
  type UploadReceipt,
  type UploadReference,
  type UploadRequestOptions,
} from "./core/upload-authority.js";

// Version
export { __version__ } from "./version.js";

export { Dataset, runEval } from "./evals.js";
export type { EvalContext, EvalResult, RunEvalOptions } from "./evals.js";
