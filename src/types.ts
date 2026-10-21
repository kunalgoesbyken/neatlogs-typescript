import type { UploadAuthorityOption } from "./core/upload-authority.js";

/**
 * Span kind for categorizing instrumented operations.
 */
export type SpanKind =
  | 'WORKFLOW'
  | 'AGENT'
  | 'CHAIN'
  | 'TOOL'
  | 'RETRIEVER'
  | 'EMBEDDING'
  | 'MCP_TOOL'
  | 'GUARDRAIL'
  | 'EVALUATOR';

/**
 * Span kinds accepted by the lower-level trace() API.
 *
 * Unlike span(), trace() can represent provider-style operations that are not
 * valid decorator kinds. Keep SpanKind narrow so span({ kind: ... }) retains
 * its runtime/type-level validation while documented custom reranker, vector
 * store, and LLM traces remain assertion-free.
 */
export type TraceSpanKind = SpanKind | 'LLM' | 'RERANKER' | 'VECTOR_STORE';

/** Context supplied to an export-boundary mask callback. */
export interface MaskContext {
  /** Aborted when the callback exceeds the SDK's masking deadline. */
  signal: AbortSignal;
  /** Signal currently being prepared for export. */
  signalType: 'span' | 'log';
  /** Deadline enforced by the SDK, in milliseconds. */
  timeoutMs: number;
}

/**
 * A mask function that can redact or transform a span or log immediately
 * before export. Return null to drop the signal entirely. The optional context
 * keeps existing one-argument callbacks source-compatible.
 */
export type MaskFunction = (
  signalData: Record<string, any>,
  context?: MaskContext,
) =>
  | Record<string, any>
  | null
  | undefined
  | Promise<Record<string, any> | null | undefined>;

/**
 * A prompt message with role and content.
 */
export interface PromptMessage {
  role: string;
  content: string;
}

/**
 * Options for initializing the Neatlogs SDK.
 */
export interface InitOptions {
  /** Neatlogs API key. Falls back to NEATLOGS_API_KEY env var. */
  apiKey?: string;
  /** Fetch these A/B test settings at init and refresh in the background every 60 seconds. */
  experiments?: { keys: string[]; baseUrl?: string };
  /** Name of the workflow being traced. Defaults to process.argv[1]. */
  workflowName?: string;
  /**
   * Operator identifier — whoever is RUNNING the SDK (a developer, a service
   * account, a CI job). Attached to all spans as a resource attribute. This is
   * NOT the end-user of your application.
   *
   * Session and end-user identity are PER-REQUEST, not process-global, so they
   * are never set here. Declare them at the trace root via `trace()`/`span()`
   * options, or for pure-`wrap()` code via the request-scoped `identify()`.
   */
  userId?: string;
  /** Tags to attach to all spans. Must be string array. */
  tags?: string[];
  /** Custom metadata to attach to all spans. */
  metadata?: Record<string, any>;
  /** Enable debug logging. Defaults to false. */
  debug?: boolean;
  /** Disable export to Neatlogs backend. Defaults to false. */
  disableExport?: boolean;
  /**
   * Capture the final normalized and masked export envelope without sending it
   * over the network. Used by the read-only local Doctor runtime.
   * @internal
   */
  diagnosticCapture?: boolean;
  /**
   * Mark the SDK Doctor's controlled OTLP export. This only adds the versioned
   * Doctor resource attributes and request header; it never changes auth,
   * tenancy, sampling, or the normal trace pipeline.
   * @internal
   */
  doctorProbe?: boolean;
  /** Test-only transport override for the controlled Doctor export. @internal */
  doctorProbeExporter?: import('@opentelemetry/sdk-trace-base').SpanExporter;
  /** Bound the controlled Doctor OTLP request. @internal */
  doctorProbeTimeoutMillis?: number;
  /**
   * Optional caller-owned private provider. Neatlogs never registers it
   * globally. The SDK adds its processors and flushes it, but never shuts it
   * down.
   */
  tracerProvider?: import('@opentelemetry/sdk-trace-base').BasicTracerProvider;
  /**
   * Register process signal handlers (beforeExit/SIGTERM/SIGINT) that flush and
   * shut down the SDK on exit. Defaults to true whenever Neatlogs owns the tracer
   * private provider, so standalone scripts drain their spans. Defaults to
   * false only when you supply your own `tracerProvider`, since its owner
   * controls that lifecycle.
   */
  registerShutdownHandlers?: boolean;
  /** Global mask function applied to all spans. */
  mask?: MaskFunction;
  /** Sampling rate (0.0 to 1.0). Defaults to 1.0. */
  sampleRate?: number;
  /** Whether to capture log records. Defaults to false. */
  captureLogs?: boolean;
  /** PII detection settings. */
  pii?: 'redact' | 'hash' | false;
  /** SDK version override. */
  version?: string;
  /** Base ingest endpoint. Defaults to https://ingest.neatlogs.com. */
  endpoint?: string;
  /** Maximum spans per export batch. Defaults to 100. */
  batchSize?: number;
  /** Seconds between batch flushes. Defaults to 5. */
  flushInterval?: number;
  /** Override team-level PII redaction toggle. true = enable, false = disable. */
  piiEnabled?: boolean;
  /** Override which span types have server-side PII redaction applied. */
  piiSpanTypes?: string[];
  /**
   * Enable the authenticated Phase 8 typed-media and OTLP overflow authority,
   * or inject a compatible authority. Defaults to NEATLOGS_UPLOADS_ENABLED and
   * remains disabled when neither is set.
   */
  uploadAuthority?: UploadAuthorityOption;
}

/**
 * Options for the span() function wrapper.
 */
export interface SpanOptions {
  /** The kind of span. Required. */
  kind: SpanKind;
  /** Custom name for the span. Defaults to the function name. */
  name?: string;
  /** Human-readable description of what this span does. */
  description?: string;
  /** Whether to capture function input. Defaults to true. */
  captureInput?: boolean;
  /** Whether to capture function output. Defaults to true. */
  captureOutput?: boolean;
  /** Per-span mask function. */
  mask?: MaskFunction;
  /** Mark this span as internal (not user-facing). */
  internal?: boolean;

  // Agent-specific
  /** Agent role (for kind: AGENT). */
  role?: string;
  /** Agent goal (for kind: AGENT). */
  goal?: string;

  // Tool-specific
  /** Tool name (for kind: TOOL or MCP_TOOL). */
  toolName?: string;
  /** Serve saved JSON tool results inside a local ReplaySession. Promise factories must declare result: 'promise'. */
  replay?: true | { result: 'value' | 'promise' };
  /** Tool parameters schema (for kind: TOOL). */
  parameters?: Record<string, any>;
  /** JSON schema describing the tool interface (for kind: MCP_TOOL). */
  toolJsonSchema?: Record<string, any>;

  /**
   * Session this trace belongs to — groups the traces of a multi-turn
   * conversation (one turn = one trace). Usually set on the WORKFLOW root. The
   * backend groups traces by it; when absent it falls back to the trace id.
   */
  sessionId?: string;
  /** Optional immediate parent session. */
  parentSessionId?: string;
  /** Product feature that initiated this request. */
  sessionFeatureName?: string;
  /** Application entry point that initiated this request. */
  sessionEntryPoint?: string;

  // End-user identity (one end-user per trace; usually set on the WORKFLOW root)
  /**
   * Identifier of the END-USER this trace belongs to — the user of your
   * application, not the operator running the SDK. The backend rolls it up to
   * the trace + session. Distinct from `init({ userId })`.
   */
  endUserId?: string;
  /** Optional arbitrary end-user fields stored as JSON on the trace. */
  endUserMetadata?: Record<string, any>;

  // Embedding-specific
  /** Embedding model name (for kind: EMBEDDING). */
  model?: string;
  /** Embedding dimension (for kind: EMBEDDING). */
  dimension?: number;
}

/**
 * Options for the trace() context wrapper.
 */
export interface TraceOptions {
  /** Name for the trace span. */
  name: string;
  /** Span kind. Defaults to 'CHAIN'. */
  kind?: TraceSpanKind;
  /** Session this root trace belongs to. */
  sessionId?: string;
  /** Optional immediate parent session. */
  parentSessionId?: string;
  /** Product feature that initiated this request. */
  sessionFeatureName?: string;
  /** Application entry point that initiated this request. */
  sessionEntryPoint?: string;
  /** Input data for this span. Auto-serialized to input.value. */
  input?: any;
  /** Prompt template to associate with this trace (string or PromptTemplate instance). */
  promptTemplate?: string | { template: string | PromptMessage[]; variables: string[] };
  /** Prompt variables used in this trace. */
  promptVariables?: Record<string, any>;
  /** User prompt template (string or UserPromptTemplate instance). */
  userPromptTemplate?: string | { template: string | PromptMessage[]; variables: string[] };
  /** User prompt variables used in this trace. */
  userPromptVariables?: Record<string, any>;
  /** Prompt version identifier. */
  version?: string;
  /** Per-trace mask function. */
  mask?: MaskFunction;
  /**
   * Identifier of the END-USER this trace belongs to — the user of your
   * application, not the operator running the SDK. One end-user per trace; the
   * backend rolls it up to the trace + session. Distinct from `init({ userId })`.
   */
  endUserId?: string;
  /** Optional arbitrary end-user fields stored as JSON on the trace. */
  endUserMetadata?: Record<string, any>;
  /** Custom attributes to set on the span. */
  attributes?: Record<string, any>;
  /** Allow extra attributes via index signature. */
  [key: string]: any;
}

/**
 * Cached prompt data from the Neatlogs API.
 */
export interface CachedPrompt {
  id: string;
  name: string;
  version: number;
  content: string | null;
  messages: PromptMessage[] | null;
  config: Record<string, any>;
  labels: string[];
  updatedAt: string;
  type: 'text' | 'chat';
}

/**
 * Session configuration stored after init().
 */
export interface SessionConfig {
  userId?: string;
  workflowName: string;
  tags: string[];
  metadata: Record<string, any>;
  pii: 'redact' | 'hash' | false;
}
