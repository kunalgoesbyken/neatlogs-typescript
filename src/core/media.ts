import { createHash } from "node:crypto";
import { getActiveClient } from "./active-client.js";
import type { DeliveryDiagnostics } from "./delivery-diagnostics.js";
import {
  assertReadyUploadReceipt,
  DEFAULT_MAX_TYPED_MEDIA_BYTES,
  isSupportedTypedMediaMimeType,
  TelemetryUploadError,
  type UploadAuthority,
  type UploadPayload,
} from "./upload-authority.js";

export type MediaRecord = Record<string, string | number>;

export const DEFAULT_INLINE_MEDIA_BYTES = 100_000;
export const DEFAULT_MAX_PENDING_MEDIA_ITEMS = 32;
export const DEFAULT_MEDIA_EXPORT_DEADLINE_MS = 15_000;
const MAX_MEDIA_TRAVERSAL_DEPTH = 32;
const MAX_MEDIA_TRAVERSAL_NODES = 4_096;
const MAX_MEDIA_REPLACEMENT_KEY_CHARS = 1024 * 1024;
const MAX_MEDIA_REFERENCE_CHARS = 8_192;
const MAX_CAPTURED_MEDIA_REFERENCES = 64;
const MAX_ENCODED_MEDIA_CHARS = Math.ceil((DEFAULT_MAX_TYPED_MEDIA_BYTES * 4) / 3) + 8;
const REDACTED_MEDIA_STRUCTURE = "[REDACTED_MEDIA_STRUCTURE]";
const MEDIA_PAYLOAD_SIGNAL =
  /(?:data\s*:|"(?:image_url|input_audio|inline_data|inlineData|file_data|fileData|file_url|file_uri|fileUri|b64_json|mime_type|mimeType|media_type|mediaType)"\s*:|"type"\s*:\s*"(?:image|audio|video|document|file|input_file|image_generation_call)")/i;

interface DiscoveredMedia {
  record: MediaRecord;
  original?: string;
  binaryOriginal?: Uint8Array;
  bytes?: Uint8Array;
}

interface MediaDiscoveryResult {
  items: DiscoveredMedia[];
  truncated: boolean;
}

interface PendingMedia {
  bytes: Uint8Array;
  prefixes: string[];
  record: MediaRecord;
  token: string;
  sha256: string;
  byteLength: number;
  mimeType: string;
}

interface PendingMediaAccount {
  bytes: number;
  items: number;
  released: boolean;
  owner: PendingMediaOwnerState;
  span: WeakRef<object>;
}

interface PendingMediaBucket {
  items: PendingMedia[];
  account: PendingMediaAccount;
  unregisterToken: object;
}
interface PendingMediaOwnerState {
  bytes: number;
  items: number;
  accounts: Set<PendingMediaAccount>;
}

interface MediaRuntimeState {
  pendingBySpan: WeakMap<object, PendingMediaBucket>;
  pendingMediaOwners: WeakMap<object, PendingMediaOwnerState>;
  mediaCaptureAvailability: WeakMap<object, { available: boolean; reason: string }>;
  mediaCaptureOwners: WeakMap<object, object>;
  spanAliases: WeakMap<object, object>;
  defaultPendingMediaOwner: object;
}

const MEDIA_RUNTIME_KEY = Symbol.for("neatlogs.media-runtime.v1");
const runtimeGlobal = globalThis as typeof globalThis & Record<symbol, unknown>;
const mediaRuntime =
  (runtimeGlobal[MEDIA_RUNTIME_KEY] as MediaRuntimeState | undefined) ??
  ({
    pendingBySpan: new WeakMap(),
    pendingMediaOwners: new WeakMap(),
    mediaCaptureAvailability: new WeakMap(),
    mediaCaptureOwners: new WeakMap(),
    spanAliases: new WeakMap(),
    defaultPendingMediaOwner: {},
  } satisfies MediaRuntimeState);
runtimeGlobal[MEDIA_RUNTIME_KEY] = mediaRuntime;

const {
  pendingBySpan,
  pendingMediaOwners,
  mediaCaptureAvailability,
  mediaCaptureOwners,
  spanAliases,
  defaultPendingMediaOwner,
} = mediaRuntime;

function canonicalSpan(span: object): object {
  return spanAliases.get(span) ?? span;
}

/** Make a transparent span facade share media state with its SDK span. */
export function aliasMediaCaptureSpan(alias: object, target: object): void {
  spanAliases.set(alias, canonicalSpan(target));
}

/** Bind a recording SDK span to the upload gate selected by its pipeline. */
export function setMediaCaptureAvailability(
  span: object,
  available: boolean,
  reason = "telemetry_uploads_disabled",
  owner?: object,
): void {
  const target = canonicalSpan(span);
  mediaCaptureAvailability.set(target, {
    available,
    reason: safeUnavailableReason(reason),
  });
  if (owner) mediaCaptureOwners.set(target, owner);
}

function pendingOwnerState(owner: object): PendingMediaOwnerState {
  const prior = pendingMediaOwners.get(owner);
  if (prior) return prior;
  const state = {
    bytes: 0,
    items: 0,
    accounts: new Set<PendingMediaAccount>(),
  };
  pendingMediaOwners.set(owner, state);
  return state;
}

function releasePendingAccount(account: PendingMediaAccount): void {
  if (account.released) return;
  account.released = true;
  account.owner.bytes = Math.max(0, account.owner.bytes - account.bytes);
  account.owner.items = Math.max(0, account.owner.items - account.items);
  account.owner.accounts.delete(account);
}

const pendingFinalizer =
  typeof FinalizationRegistry === "undefined"
    ? undefined
    : new FinalizationRegistry<PendingMediaAccount>(releasePendingAccount);

function pendingBucket(span: object): PendingMediaBucket {
  span = canonicalSpan(span);
  const prior = pendingBySpan.get(span);
  if (prior) return prior;
  const owner = pendingOwnerState(
    mediaCaptureOwners.get(span) ?? getActiveClient() ?? defaultPendingMediaOwner,
  );
  const account: PendingMediaAccount = {
    bytes: 0,
    items: 0,
    released: false,
    owner,
    span: new WeakRef(span),
  };
  const unregisterToken = {};
  const bucket = { items: [], account, unregisterToken };
  owner.accounts.add(account);
  pendingBySpan.set(span, bucket);
  pendingFinalizer?.register(span, account, unregisterToken);
  return bucket;
}

function detachPendingBucket(span: object): PendingMediaBucket | undefined {
  const alias = span;
  span = canonicalSpan(alias);
  spanAliases.delete(alias);
  const bucket = pendingBySpan.get(span);
  if (!bucket) return undefined;
  pendingBySpan.delete(span);
  mediaCaptureAvailability.delete(span);
  mediaCaptureOwners.delete(span);
  pendingFinalizer?.unregister(bucket.unregisterToken);
  return bucket;
}

function uploadToken(record: MediaRecord): string {
  const material = `${record.sha256}:${record.mime_type}:${record.purpose}`;
  return `nl_pending_media_${createHash("sha256").update(material).digest("hex").slice(0, 24)}`;
}

function mediaKind(mimeType: string, declared = ""): string {
  const value = declared
    .toLowerCase()
    .replace(/^(?:input|output)_/, "")
    .replace(/_url$/, "");
  if (value === "image_generation_call") return "image";
  if (value === "file") return "document";
  if (["image", "audio", "video", "document"].includes(value)) return value;
  if (mimeType.startsWith("image/")) return "image";
  if (mimeType.startsWith("audio/")) return "audio";
  if (mimeType.startsWith("video/")) return "video";
  if (mimeType === "application/pdf" || mimeType.startsWith("text/")) {
    return "document";
  }
  return "media";
}

function declaredMediaKind(value: string): string {
  const normalized = mediaKind("", value);
  return normalized === "media" ? "" : normalized;
}

function canonicalMimeType(value: string): string {
  const mimeType = value.trim().split(";", 1)[0]?.toLowerCase() ?? "";
  return mimeType === "image/jpg"
    ? "image/jpeg"
    : mimeType === "audio/mp3"
      ? "audio/mpeg"
      : mimeType;
}

interface DecodedBase64 {
  bytes?: Uint8Array;
  byteLength: number;
  sha256: string;
}

function rejectedMediaRecord(
  original: string,
  mimeType: string,
  declared: string,
  purpose: string,
  reason: string,
  source: "inline" | "url",
): DiscoveredMedia {
  const fingerprint = createHash("sha256")
    .update(original.slice(0, 4096))
    .update(String(original.length))
    .digest("hex");
  return {
    original,
    record: {
      id: `nl_media_rejected_${fingerprint.slice(0, 24)}`,
      type: mediaKind(mimeType, declared),
      source,
      mime_type: mimeType || "application/octet-stream",
      purpose,
      state: "failed",
      safe_preview: `upload failed: validate/${reason}`,
    },
  };
}

function decodeBase64(value: string, retainBytes = true): DecodedBase64 | null {
  const normalized = value.replace(/\s+/g, "");
  if (!/^[A-Za-z0-9+/]*={0,2}$/.test(normalized) || normalized.length % 4 === 1) {
    return null;
  }
  try {
    const padding = normalized.endsWith("==") ? 2 : normalized.endsWith("=") ? 1 : 0;
    const estimatedBytes = Math.floor((normalized.length * 3) / 4) - padding;
    if (retainBytes && estimatedBytes <= DEFAULT_MAX_TYPED_MEDIA_BYTES) {
      const bytes = Buffer.from(normalized, "base64");
      if (bytes.toString("base64").replace(/=+$/, "") !== normalized.replace(/=+$/, "")) {
        return null;
      }
      return {
        bytes,
        byteLength: bytes.byteLength,
        sha256: createHash("sha256").update(bytes).digest("hex"),
      };
    }
    const digest = createHash("sha256");
    let byteLength = 0;
    const chunkCharacters = 64 * 1024;
    for (let offset = 0; offset < normalized.length; offset += chunkCharacters) {
      const chunk = Buffer.from(normalized.slice(offset, offset + chunkCharacters), "base64");
      digest.update(chunk);
      byteLength += chunk.byteLength;
    }
    if (byteLength !== estimatedBytes) return null;
    return { byteLength, sha256: digest.digest("hex") };
  } catch {
    return null;
  }
}

function inlineRecord(
  input: string | Uint8Array,
  mimeType: string,
  declared: string,
  purpose: string,
  retainBytes = true,
): DiscoveredMedia | null {
  mimeType = canonicalMimeType(mimeType);
  let original: string | undefined;
  let binaryOriginal: Uint8Array | undefined;
  let bytes: Uint8Array | undefined;
  let byteLength: number;
  let sha256: string;
  if (typeof input === "string") {
    original = input;
    let encoded = input.trim();
    if (/^data:/i.test(encoded)) {
      const comma = encoded.indexOf(",");
      if (comma < 0) return null;
      const header = encoded.slice(5, comma);
      if (!header.toLowerCase().split(";").includes("base64")) return null;
      mimeType = canonicalMimeType(header.split(";", 1)[0] || mimeType);
      encoded = encoded.slice(comma + 1);
    }
    if (encoded.length > MAX_ENCODED_MEDIA_CHARS) {
      return rejectedMediaRecord(
        original,
        mimeType,
        declared,
        purpose,
        "payload_too_large",
        "inline",
      );
    }
    const decoded = decodeBase64(encoded, retainBytes);
    if (!decoded) {
      return rejectedMediaRecord(
        original,
        mimeType,
        declared,
        purpose,
        "invalid_inline_media",
        "inline",
      );
    }
    bytes = decoded.bytes;
    byteLength = decoded.byteLength;
    sha256 = decoded.sha256;
  } else {
    binaryOriginal = input;
    byteLength = input.byteLength;
    sha256 = createHash("sha256").update(input).digest("hex");
    if (retainBytes && byteLength <= DEFAULT_MAX_TYPED_MEDIA_BYTES) {
      bytes = Uint8Array.from(input);
    }
  }
  const eligible =
    byteLength > DEFAULT_INLINE_MEDIA_BYTES && byteLength <= DEFAULT_MAX_TYPED_MEDIA_BYTES;
  const tooLarge = byteLength > DEFAULT_MAX_TYPED_MEDIA_BYTES;
  return {
    original,
    binaryOriginal,
    bytes,
    record: {
      id: `nl_media_${sha256.slice(0, 24)}`,
      type: mediaKind(mimeType, declared),
      source: "inline",
      mime_type: mimeType || "application/octet-stream",
      byte_length: byteLength,
      sha256,
      purpose,
      state: tooLarge ? "failed" : eligible ? "pending-upload" : "inline",
      ...(tooLarge
        ? { safe_preview: "upload failed: validate/payload_too_large" }
        : eligible
          ? { safe_preview: "awaiting authenticated upload" }
          : {}),
    },
  };
}

function referenceRecord(
  reference: string,
  mimeType: string,
  declared: string,
  purpose: string,
): DiscoveredMedia | null {
  mimeType = canonicalMimeType(mimeType);
  if (reference.length > MAX_MEDIA_REFERENCE_CHARS) {
    return rejectedMediaRecord(
      reference,
      mimeType,
      declared,
      purpose,
      "media_reference_too_large",
      "url",
    );
  }
  let safeReference = reference;
  let isURLReference = false;
  const normalizedReference = reference.trim();
  const protocolRelative = normalizedReference.startsWith("//");
  const relativeReference =
    !protocolRelative &&
    !/^[a-z][a-z\d+.-]*:/i.test(normalizedReference) &&
    (normalizedReference.includes("/") ||
      normalizedReference.includes("?") ||
      normalizedReference.includes("#"));
  try {
    const parsed = protocolRelative || relativeReference
      ? new URL(normalizedReference, "https://neatlogs.invalid")
      : new URL(normalizedReference);
    if (parsed.protocol.toLowerCase() === "data:") {
      return rejectedMediaRecord(
        reference,
        mimeType,
        declared,
        purpose,
        "invalid_inline_media",
        "inline",
      );
    }
    const hierarchical =
      protocolRelative ||
      relativeReference ||
      parsed.protocol === "http:" ||
      parsed.protocol === "https:" ||
      /^[a-z][a-z\d+.-]*:\/\//i.test(normalizedReference) ||
      parsed.pathname.startsWith("/");
    if (hierarchical) {
      isURLReference = true;
      parsed.username = "";
      parsed.password = "";
      parsed.search = "";
      parsed.hash = "";
      safeReference = protocolRelative
        ? `//${parsed.host}${parsed.pathname}`
        : relativeReference
          ? parsed.pathname
          : parsed.toString();
    } else if (parsed.protocol) {
      isURLReference = true;
      safeReference = parsed.protocol;
    }
  } catch {
    if (/^(?:\/\/|[a-z][a-z\d+.-]*:)/i.test(normalizedReference)) {
      return rejectedMediaRecord(
        reference,
        mimeType,
        declared,
        purpose,
        "invalid_media_reference",
        "url",
      );
    }
  }
  const digest = createHash("sha256").update(reference).digest("hex");
  return {
    original: reference,
    record: {
      id: `nl_media_${digest.slice(0, 24)}`,
      type: mediaKind(mimeType, declared),
      source: isURLReference ? "url" : "provider",
      mime_type: mimeType || "application/octet-stream",
      reference: safeReference,
      purpose,
      state: "available",
    },
  };
}

function discoverMedia(
  value: unknown,
  purpose: string,
  retainBytes = true,
): MediaDiscoveryResult {
  const found: DiscoveredMedia[] = [];
  const discoveredKeys = new Set<string>();
  let retainedBytes = 0;
  let truncated = false;
  const append = (item: DiscoveredMedia | null): void => {
    if (!item) return;
    if (found.length >= MAX_CAPTURED_MEDIA_REFERENCES) {
      truncated = true;
      return;
    }
    const key = `${item.record.id}:${item.record.mime_type}:${item.record.reference ?? ""}:${item.record.type}`;
    if (discoveredKeys.has(key)) return;
    discoveredKeys.add(key);
    if (item.bytes && item.record.state === "pending-upload") {
      if (retainedBytes + item.bytes.byteLength > DEFAULT_MAX_TYPED_MEDIA_BYTES) {
        item.bytes = undefined;
        item.record.state = "failed";
        item.record.safe_preview = "upload failed: validate/pending_media_memory_limit";
      } else {
        retainedBytes += item.bytes.byteLength;
      }
    }
    found.push(item);
  };
  const visited = new WeakSet<object>();
  let visitedNodes = 0;
  const visit = (node: any, inheritedDeclared = "", inheritedMimeType = "", depth = 0): void => {
    if (typeof node === "string" && /^\s*data:/i.test(node)) {
      append(inlineRecord(node, inheritedMimeType, inheritedDeclared, purpose, retainBytes));
      return;
    }
    if (!node || typeof node !== "object") return;
    if (found.length >= MAX_CAPTURED_MEDIA_REFERENCES) {
      truncated = true;
      return;
    }
    if (depth > MAX_MEDIA_TRAVERSAL_DEPTH || visitedNodes >= MAX_MEDIA_TRAVERSAL_NODES) {
      return;
    }
    if (visited.has(node)) return;
    visited.add(node);
    visitedNodes += 1;
    try {
      if (node instanceof Uint8Array) return;
      if (Array.isArray(node)) {
        for (const item of node) {
          if (visitedNodes >= MAX_MEDIA_TRAVERSAL_NODES) break;
          if (inheritedDeclared && (typeof item === "string" || item instanceof Uint8Array)) {
            const record =
              typeof item === "string" && /^(?:https?:|s3:|gs:|\/\/)/i.test(item)
                ? referenceRecord(item, inheritedMimeType, inheritedDeclared, purpose)
                : inlineRecord(item, inheritedMimeType, inheritedDeclared, purpose, retainBytes);
            append(record);
          } else {
            visit(item, inheritedDeclared, inheritedMimeType, depth + 1);
          }
        }
        return;
      }
      const entries: [string, unknown][] = Object.entries(node);
      const read = (key: string): unknown => entries.find(([name]) => name === key)?.[1];
      const declared = String(read("type") ?? inheritedDeclared);
      const normalizedDeclared = declaredMediaKind(declared);
      const mediaDeclared = normalizedDeclared || declaredMediaKind(inheritedDeclared);
      let mimeType = canonicalMimeType(
        String(
          read("mime_type") ??
            read("mimeType") ??
            read("media_type") ??
            read("mediaType") ??
            inheritedMimeType,
        ),
      );
      const declaredFormat = read("format") ?? read("output_format") ?? read("outputFormat");
      if (!mimeType && mediaDeclared && typeof declaredFormat === "string") {
        const format = declaredFormat.toLowerCase();
        mimeType = canonicalMimeType(
          mediaDeclared === "document" && format === "pdf"
            ? "application/pdf"
            : `${mediaDeclared}/${format}`,
        );
      }
      const imageUrl = read("image_url");
      let image = imageUrl;
      if (imageUrl && typeof imageUrl === "object") {
        try {
          image = (imageUrl as Record<string, unknown>).url;
        } catch {
          image = undefined;
        }
      }
      if (typeof image === "string") {
        const record = /^\s*data:/i.test(image)
          ? inlineRecord(image, mimeType, "image", purpose, retainBytes)
          : referenceRecord(image, mimeType, "image", purpose);
        append(record);
      }
      const audio = read("input_audio") ?? read("inputAudio");
      if (audio && typeof audio === "object" && typeof (audio as any).data === "string") {
        const format = String((audio as any).format ?? "unknown");
        const record = inlineRecord(
          (audio as any).data,
          mimeType || `audio/${format}`,
          "audio",
          purpose,
          retainBytes,
        );
        append(record);
      }
      const inline = read("inline_data") ?? read("inlineData");
      if (inline && typeof inline === "object" && typeof (inline as any).data === "string") {
        const record = inlineRecord(
          (inline as any).data,
          String((inline as any).mime_type ?? (inline as any).mimeType ?? mimeType),
          mediaDeclared,
          purpose,
          retainBytes,
        );
        append(record);
      }
      const fileData = read("file_data") ?? read("fileData");
      if (typeof fileData === "string") {
        const record = inlineRecord(
          fileData,
          mimeType,
          mediaDeclared || "document",
          purpose,
          retainBytes,
        );
        append(record);
      }
      const raw =
        read("data") ??
        read("bytes") ??
        read("b64_json") ??
        (normalizedDeclared === "image" ? read("result") : undefined);
      if ((typeof raw === "string" || raw instanceof Uint8Array) && (mediaDeclared || mimeType)) {
        const record = inlineRecord(raw, mimeType, mediaDeclared, purpose, retainBytes);
        append(record);
      }
      const reference =
        read("file_id") ?? read("file_url") ?? read("file_uri") ?? read("fileUri") ?? read("url");
      if (
        typeof reference === "string" &&
        (["file", "url"].includes(normalizedDeclared) || !!mediaDeclared || !!mimeType)
      ) {
        const record = /^\s*data:/i.test(reference)
          ? inlineRecord(reference, mimeType, mediaDeclared || "document", purpose, retainBytes)
          : referenceRecord(reference, mimeType, mediaDeclared || "document", purpose);
        append(record);
      }
      for (const [key, item] of entries) {
        if (visitedNodes >= MAX_MEDIA_TRAVERSAL_NODES) break;
        const keyed = key
          .toLowerCase()
          .replace(/^input_/, "")
          .replace(/s$/, "");
        const childDeclared = declaredMediaKind(keyed) || mediaDeclared;
        let childMimeType = mimeType;
        if (!childMimeType && childDeclared && typeof declaredFormat === "string") {
          const format = declaredFormat.toLowerCase();
          childMimeType = canonicalMimeType(
            childDeclared === "document" && format === "pdf"
              ? "application/pdf"
              : `${childDeclared}/${format}`,
          );
        }
        visit(item, childDeclared, childMimeType, depth + 1);
      }
    } catch {
      return;
    }
  };
  visit(value);
  return { items: found, truncated };
}

function placeholder(record: MediaRecord): Record<string, unknown> {
  return {
    neatlogs_media: Object.fromEntries(
      Object.entries(record).filter(([key]) =>
        [
          "id",
          "type",
          "source",
          "mime_type",
          "byte_length",
          "sha256",
          "purpose",
          "state",
          "reference",
          "content_encoding",
          "safe_preview",
          "upload_token",
        ].includes(key),
      ),
    ),
  };
}

function sanitizedCopy(value: unknown, discovered: DiscoveredMedia[], markLossy: () => void = () => undefined): unknown {
  const replacements = new Map<string, unknown>();
  const largeReplacements: Array<[string, unknown]> = [];
  for (const item of discovered) {
    if (!item.original) continue;
    if (item.record.state === "pending-upload" || item.record.state === "failed") {
      const replacement = placeholder(item.record);
      if (item.original.length > MAX_MEDIA_REPLACEMENT_KEY_CHARS) {
        largeReplacements.push([item.original, replacement]);
      } else {
        replacements.set(item.original, replacement);
      }
    } else if (typeof item.record.reference === "string") {
      if (item.original.length > MAX_MEDIA_REPLACEMENT_KEY_CHARS) {
        largeReplacements.push([item.original, item.record.reference]);
      } else {
        replacements.set(item.original, item.record.reference);
      }
    }
  }
  const visited = new WeakMap<object, unknown>();
  let visitedNodes = 0;
  const clone = (node: any, depth = 0): any => {
    if (typeof node === "string") {
      const direct =
        node.length <= MAX_MEDIA_REPLACEMENT_KEY_CHARS ? replacements.get(node) : undefined;
      if (direct !== undefined) { if (direct !== node) markLossy(); return direct; }
      const replacement = largeReplacements.find(([original]) => original === node)?.[1];
      if (replacement !== undefined && replacement !== node) markLossy();
      return replacement ?? node;
    }
    if (node instanceof Uint8Array) {
      const match = discovered.find(
        (item) =>
          item.binaryOriginal === node ||
          (item.bytes?.buffer === node.buffer && item.bytes.byteOffset === node.byteOffset),
      );
      if (match && ["pending-upload", "failed"].includes(String(match.record.state))) {
        markLossy();
        return placeholder(match.record);
      }
      return node;
    }
    if (!node || typeof node !== "object") return node;
    if (depth > MAX_MEDIA_TRAVERSAL_DEPTH || visitedNodes >= MAX_MEDIA_TRAVERSAL_NODES) {
      markLossy();
      return REDACTED_MEDIA_STRUCTURE;
    }
    const prior = visited.get(node);
    if (prior) { markLossy(); return "[CIRCULAR]"; }
    visitedNodes += 1;
    if (Array.isArray(node)) {
      const result: unknown[] = [];
      visited.set(node, result);
      for (const item of node) {
        if (visitedNodes >= MAX_MEDIA_TRAVERSAL_NODES) {
          markLossy();
          result.push(REDACTED_MEDIA_STRUCTURE);
          break;
        }
        result.push(clone(item, depth + 1));
      }
      return result;
    }
    const result: Record<string, unknown> = {};
    visited.set(node, result);
    try {
      for (const [key, item] of Object.entries(node)) {
        if (visitedNodes >= MAX_MEDIA_TRAVERSAL_NODES) {
          markLossy();
          result.neatlogs_truncated = REDACTED_MEDIA_STRUCTURE;
          break;
        }
        result[key] = clone(item, depth + 1);
      }
    } catch {
      markLossy();
      return REDACTED_MEDIA_STRUCTURE;
    }
    return result;
  };
  return clone(value);
}

export function mediaReferences(value: unknown, purpose: string): MediaRecord[] {
  return discoverMedia(value, purpose, false).items.map(({ record }) =>
    record.state === "pending-upload"
      ? {
          ...record,
          state: "failed",
          safe_preview: "upload unavailable: telemetry_uploads_disabled",
        }
      : { ...record },
  );
}

/** Remove large inline bytes and URL credentials from a captured value. */
export function sanitizeMediaPayload(value: unknown, purpose = "capture"): unknown {
  try {
    const discovered = discoverMedia(value, purpose, false);
    return discovered.truncated
      ? REDACTED_MEDIA_STRUCTURE
      : sanitizedCopy(value, discovered.items);
  } catch {
    return REDACTED_MEDIA_STRUCTURE;
  }
}

export interface CapturedMediaValue {
  value: unknown;
  count: number;
}

/** Capture typed media metadata from a stable index and return a telemetry-safe clone. */
export function captureMediaWithIndex(
  span: {
    setAttribute(name: string, value: string | number): unknown;
    isRecording?(): boolean;
  },
  prefix: string,
  value: unknown,
  purpose: string,
  mediaIndexOffset = 0,
): CapturedMediaValue {
  let recording = false;
  try {
    recording = typeof span.isRecording !== "function" || span.isRecording();
  } catch {
    span.setAttribute(`${prefix}.media.capture_error`, "media_capture_failed");
    span.setAttribute('neatlogs.capture_fidelity', 'unavailable');
    return { value: REDACTED_MEDIA_STRUCTURE, count: 0 };
  }
  const spanKey = canonicalSpan(span as object);
  const gate = mediaCaptureAvailability.get(spanKey);
  const uploadsAvailable = gate?.available ?? true;
  let discovery: MediaDiscoveryResult;
  try {
    discovery = discoverMedia(value, purpose, recording && uploadsAvailable);
  } catch {
    span.setAttribute(`${prefix}.media.capture_error`, "media_capture_failed");
    span.setAttribute('neatlogs.capture_fidelity', 'unavailable');
    return { value: REDACTED_MEDIA_STRUCTURE, count: 0 };
  }
  discovery.items.forEach((item, index) => {
    const recordPrefix = `${prefix}.media.${mediaIndexOffset + index}`;
    if (item.record.state === "pending-upload") {
      if (!recording) {
        item.bytes = undefined;
        item.record.state = "failed";
        item.record.safe_preview = "upload unavailable: span_not_recording";
      } else if (!uploadsAvailable) {
        item.bytes = undefined;
        item.record.state = "failed";
        item.record.safe_preview = `upload unavailable: ${gate?.reason ?? "telemetry_uploads_disabled"}`;
      } else if (!item.bytes) {
        item.record.state = "failed";
        item.record.safe_preview = "upload failed: validate/staged_payload_missing";
      } else {
        const token = uploadToken(item.record);
        item.record.upload_token = token;
        const bucket = pendingBucket(spanKey);
        const duplicate = bucket.items.find(
          (candidate) =>
            candidate.sha256 === item.record.sha256 && candidate.mimeType === item.record.mime_type,
        );
        if (duplicate) {
          if (!duplicate.prefixes.includes(recordPrefix)) {
            duplicate.prefixes.push(recordPrefix);
          }
        } else if (bucket.account.owner.items >= DEFAULT_MAX_PENDING_MEDIA_ITEMS) {
          item.bytes = undefined;
          delete item.record.upload_token;
          item.record.state = "failed";
          item.record.safe_preview = "upload failed: validate/pending_media_item_limit";
        } else if (
          bucket.account.owner.bytes + item.bytes.byteLength >
          DEFAULT_MAX_TYPED_MEDIA_BYTES
        ) {
          item.bytes = undefined;
          delete item.record.upload_token;
          item.record.state = "failed";
          item.record.safe_preview = "upload failed: validate/pending_media_memory_limit";
        } else {
          bucket.account.owner.bytes += item.bytes.byteLength;
          bucket.account.owner.items += 1;
          bucket.account.bytes += item.bytes.byteLength;
          bucket.account.items += 1;
          bucket.items.push({
            bytes: item.bytes,
            prefixes: [recordPrefix],
            record: { ...item.record },
            token,
            sha256: String(item.record.sha256),
            byteLength: Number(item.record.byte_length),
            mimeType: String(item.record.mime_type),
          });
        }
        if (bucket.items.length === 0) {
          pendingBySpan.delete(span as object);
          pendingFinalizer?.unregister(bucket.unregisterToken);
          releasePendingAccount(bucket.account);
        }
      }
    }
    Object.entries(item.record).forEach(([key, field]) =>
      span.setAttribute(`${recordPrefix}.${key}`, field),
    );
  });
  const markLossy = () => span.setAttribute('neatlogs.capture_fidelity', 'truncated');
  if (discovery.truncated) markLossy();
  return {
    value: discovery.truncated
      ? REDACTED_MEDIA_STRUCTURE
      : sanitizedCopy(value, discovery.items, markLossy),
    count: discovery.items.length,
  };
}

/** Capture typed media metadata and return a telemetry-safe clone of the value. */
export function captureMedia(
  span: {
    setAttribute(name: string, value: string | number): unknown;
    isRecording?(): boolean;
  },
  prefix: string,
  value: unknown,
  purpose: string,
): unknown {
  return captureMediaWithIndex(span, prefix, value, purpose).value;
}

export function setMediaAttributes(
  span: { setAttribute(name: string, value: string | number): unknown },
  prefix: string,
  value: unknown,
  purpose: string,
): void {
  captureMedia(span, prefix, value, purpose);
}

/** Sanitize media that reached a completed span through a generic integration. */
export function captureMediaInSnapshot(
  span: object,
  snapshot: Record<string, any>,
  uploadsAvailable: boolean,
  unavailableReason = "telemetry_uploads_disabled",
): void {
  setMediaCaptureAvailability(span, uploadsAvailable, unavailableReason);
  const spanAttributes =
    snapshot.attributes && typeof snapshot.attributes === "object"
      ? snapshot.attributes
      : (snapshot.attributes = {});
  const setter = {
    isRecording: () => true,
    setAttribute(name: string, value: string | number) {
      spanAttributes[name] = value;
    },
  };
  aliasMediaCaptureSpan(setter, span);
  let mediaIndex = 0;

  const sanitizeAttributes = (attributes: unknown): void => {
    if (!attributes || typeof attributes !== "object" || Array.isArray(attributes)) return;
    for (const [key, original] of Object.entries(attributes)) {
      let value: unknown = original;
      let parsedJson = false;
      if (typeof original === "string") {
        if (!MEDIA_PAYLOAD_SIGNAL.test(original)) continue;
        const trimmed = original.trimStart();
        if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
          try {
            value = JSON.parse(original);
            parsedJson = true;
          } catch {
            if (/data\s*:\s*(?:image|audio|video|application\/pdf)/i.test(original)) {
              (attributes as Record<string, unknown>)[key] = REDACTED_MEDIA_STRUCTURE;
            }
            continue;
          }
        }
      }

      const purpose = key.includes("output") ? "output" : "input";
      let safeValue: unknown;
      let count = 0;
      const sanitizeOnly = mediaIndex >= MAX_CAPTURED_MEDIA_REFERENCES;
      if (sanitizeOnly) {
        safeValue = sanitizeMediaPayload(value, purpose);
      } else {
        const captured = captureMediaWithIndex(
          setter,
          "neatlogs.content",
          value,
          purpose,
          mediaIndex,
        );
        safeValue = captured.value;
        count = captured.count;
      }
      mediaIndex += count;
      if (count === 0 && !sanitizeOnly) continue;
      const replacement =
        typeof original === "string" && (parsedJson || typeof safeValue !== "string")
          ? JSON.stringify(safeValue)
          : safeValue;
      if (replacement === original) continue;
      (attributes as Record<string, unknown>)[key] = replacement;
    }
  };

  sanitizeAttributes(spanAttributes);
  sanitizeAttributes(snapshot.resource?.attributes ?? snapshot.resource);
  if (Array.isArray(snapshot.events)) {
    for (const event of snapshot.events) sanitizeAttributes(event?.attributes);
  }
  if (Array.isArray(snapshot.links)) {
    for (const link of snapshot.links) sanitizeAttributes(link?.attributes);
  }
}

function safeFailure(error: unknown): { stage: string; reason: string } {
  if (!(error instanceof TelemetryUploadError)) {
    return { stage: "upload", reason: "unexpected_error" };
  }
  return {
    stage: error.stage,
    reason: /^[a-zA-Z0-9_.-]{1,64}$/.test(error.reasonCode)
      ? error.reasonCode
      : "unsafe_reason_code",
  };
}

function safeUnavailableReason(value: string): string {
  return /^[a-zA-Z0-9_.-]{1,64}$/.test(value) ? value : "upload_authority_unavailable";
}

function rewritePendingPlaceholder(
  attributes: Record<string, any>,
  item: PendingMedia,
  update: MediaRecord,
): void {
  const matches = (value: Record<string, any>): boolean =>
    value.upload_token === item.token ||
    (value.id === item.record.id && value.sha256 === item.sha256);
  const applyUpdate = (value: Record<string, any>): Record<string, any> => {
    const after = { ...value, ...update };
    delete after.upload_token;
    if (after.state === "available") delete after.safe_preview;
    return after;
  };
  const seen = new WeakSet<object>();
  let visitedNodes = 0;
  const rewrite = (value: any, depth = 0): any => {
    if (typeof value === "string") {
      if (!value.includes(item.token) && !value.includes(String(item.record.id))) {
        return value;
      }
      if (value.length > MAX_MEDIA_REPLACEMENT_KEY_CHARS) {
        return REDACTED_MEDIA_STRUCTURE;
      }
      const trimmed = value.trimStart();
      if (!trimmed.startsWith("{") && !trimmed.startsWith("[")) return value;
      try {
        const parsed = JSON.parse(value);
        const rewritten = rewrite(parsed);
        return rewritten === parsed ? value : JSON.stringify(rewritten);
      } catch {
        return value;
      }
    }
    if (!value || typeof value !== "object") return value;
    if (depth > MAX_MEDIA_TRAVERSAL_DEPTH || visitedNodes >= MAX_MEDIA_TRAVERSAL_NODES) {
      return REDACTED_MEDIA_STRUCTURE;
    }
    if (seen.has(value)) return "[CIRCULAR]";
    seen.add(value);
    visitedNodes += 1;
    if (!Array.isArray(value) && matches(value)) return applyUpdate(value);
    let changed = false;
    const output = Array.isArray(value) ? [...value] : { ...value };
    for (const [key, child] of Object.entries(value)) {
      if (visitedNodes >= MAX_MEDIA_TRAVERSAL_NODES) {
        return REDACTED_MEDIA_STRUCTURE;
      }
      const rewritten = rewrite(child, depth + 1);
      if (rewritten !== child) {
        (output as Record<string, any>)[key] = rewritten;
        changed = true;
      }
    }
    if (!Array.isArray(value)) {
      for (const prefix of item.prefixes) {
        const tokenKey = `${prefix}.upload_token`;
        const matchesFlat =
          value[tokenKey] === item.token ||
          (value[`${prefix}.id`] === item.record.id && value[`${prefix}.sha256`] === item.sha256);
        if (!matchesFlat) continue;
        delete (output as Record<string, any>)[tokenKey];
        for (const [field, fieldValue] of Object.entries(update)) {
          (output as Record<string, any>)[`${prefix}.${field}`] = fieldValue;
        }
        if (update.state === "available") {
          delete (output as Record<string, any>)[`${prefix}.safe_preview`];
        }
        changed = true;
      }
    }
    return changed ? output : value;
  };

  for (const [key, value] of Object.entries(attributes)) {
    attributes[key] = rewrite(value);
  }
  for (const prefix of item.prefixes) {
    const tokenKey = `${prefix}.upload_token`;
    const matchesFlat =
      attributes[tokenKey] === item.token ||
      (attributes[`${prefix}.id`] === item.record.id &&
        attributes[`${prefix}.sha256`] === item.sha256);
    if (!matchesFlat) continue;
    delete attributes[tokenKey];
    for (const [field, fieldValue] of Object.entries(update)) {
      attributes[`${prefix}.${field}`] = fieldValue;
    }
    if (update.state === "available") delete attributes[`${prefix}.safe_preview`];
  }
}

function mediaUploadTokens(value: unknown, requirePendingState: boolean): Set<string> {
  const tokens = new Set<string>();
  const seen = new WeakSet<object>();
  let visitedNodes = 0;
  const visit = (node: any, depth = 0): void => {
    if (typeof node === "string") {
      if (!node.includes("nl_pending_media_")) return;
      if (!requirePendingState && /^nl_pending_media_[0-9a-f]{24}$/.test(node)) {
        tokens.add(node);
      }
      const trimmed = node.trimStart();
      if (!trimmed.startsWith("{") && !trimmed.startsWith("[")) return;
      try {
        visit(JSON.parse(node), depth + 1);
      } catch {
        // Invalid JSON cannot authorize an out-of-band upload.
      }
      return;
    }
    if (!node || typeof node !== "object" || seen.has(node)) return;
    if (depth > MAX_MEDIA_TRAVERSAL_DEPTH || visitedNodes >= MAX_MEDIA_TRAVERSAL_NODES) {
      return;
    }
    seen.add(node);
    visitedNodes += 1;
    if (
      !Array.isArray(node) &&
      (!requirePendingState || node.state === "pending-upload") &&
      typeof node.upload_token === "string"
    ) {
      tokens.add(node.upload_token);
    }
    if (!Array.isArray(node)) {
      for (const [key, token] of Object.entries(node)) {
        if (!key.endsWith(".upload_token") || typeof token !== "string") continue;
        const prefix = key.slice(0, -".upload_token".length);
        if (!requirePendingState || node[`${prefix}.state`] === "pending-upload") {
          tokens.add(token);
        }
      }
    }
    try {
      for (const child of Object.values(node)) {
        if (visitedNodes >= MAX_MEDIA_TRAVERSAL_NODES) break;
        visit(child, depth + 1);
      }
    } catch {
      return;
    }
  };
  visit(value);
  return tokens;
}

function scrubUploadTokens(value: unknown): unknown {
  const seen = new WeakMap<object, unknown>();
  let visitedNodes = 0;
  const rewrite = (node: any, depth = 0): any => {
    if (typeof node === "string") {
      if (!node.includes("nl_pending_media_")) return node;
      const trimmed = node.trimStart();
      if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
        try {
          if (node.length > MAX_MEDIA_REPLACEMENT_KEY_CHARS) {
            return REDACTED_MEDIA_STRUCTURE;
          }
          return JSON.stringify(rewrite(JSON.parse(node), depth + 1));
        } catch {
          // Fall through to remove any opaque token embedded in plain text.
        }
      }
      return node.replace(/nl_pending_media_[0-9a-f]{24}/g, "[REDACTED_UPLOAD_TOKEN]");
    }
    if (!node || typeof node !== "object") return node;
    if (depth > MAX_MEDIA_TRAVERSAL_DEPTH || visitedNodes >= MAX_MEDIA_TRAVERSAL_NODES) {
      return REDACTED_MEDIA_STRUCTURE;
    }
    const prior = seen.get(node);
    if (prior) return "[CIRCULAR]";
    visitedNodes += 1;
    const output: any = Array.isArray(node) ? [] : {};
    seen.set(node, output);
    try {
      for (const [key, child] of Object.entries(node)) {
        if (visitedNodes >= MAX_MEDIA_TRAVERSAL_NODES) {
          return REDACTED_MEDIA_STRUCTURE;
        }
        if (key === "upload_token" || key.endsWith(".upload_token")) continue;
        output[key] = rewrite(child, depth + 1);
      }
    } catch {
      return REDACTED_MEDIA_STRUCTURE;
    }
    return output;
  };
  return rewrite(value);
}

async function uploadWithinDeadline(
  authority: UploadAuthority,
  payload: UploadPayload,
  deadlineUnixMs: number,
) {
  const remaining = deadlineUnixMs - Date.now();
  if (remaining <= 0) {
    throw new TelemetryUploadError("prepare", "media_export_deadline_exceeded", true);
  }
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      authority.upload(payload, { deadlineUnixMs }),
      new Promise<never>((_resolve, reject) => {
        timer = setTimeout(
          () => reject(new TelemetryUploadError("prepare", "media_export_deadline_exceeded", true)),
          remaining,
        );
        timer.unref?.();
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

/** Upload only media references retained by the accepted masked snapshot. */
export async function resolvePendingMediaUploads(
  span: object,
  attributes: Record<string, any>,
  authority: UploadAuthority,
  diagnostics?: DeliveryDiagnostics,
): Promise<boolean> {
  const bucket = detachPendingBucket(span);
  const pending = bucket?.items ?? [];
  const deadlineUnixMs = Date.now() + DEFAULT_MEDIA_EXPORT_DEADLINE_MS;
  let failed = false;
  try {
    const retainedTokens = mediaUploadTokens(attributes, true);
    const presentTokens = mediaUploadTokens(attributes, false);
    for (const item of pending) {
      const retained = item.prefixes.filter(
        (prefix) =>
          attributes[`${prefix}.upload_token`] === item.token &&
          attributes[`${prefix}.state`] === "pending-upload",
      );
      const retainedCount = Math.max(retained.length, retainedTokens.has(item.token) ? 1 : 0);
      if (!retainedTokens.has(item.token)) {
        const tokenWasRetainedInInvalidState = presentTokens.has(item.token);
        if (tokenWasRetainedInInvalidState) {
          failed = true;
          diagnostics?.recordTypedMedia("failures", Math.max(retainedCount, 1));
          diagnostics?.recordUploadFailure("validate", "masked_pending_state");
        }
        rewritePendingPlaceholder(attributes, item, {
          state: "failed",
          safe_preview: tokenWasRetainedInInvalidState
            ? "upload failed: validate/masked_pending_state"
            : "upload skipped: media reference removed by mask",
        });
        continue;
      }
      if (!authority.available) {
        const unavailableReason = safeUnavailableReason(authority.unavailableReason);
        failed = true;
        diagnostics?.recordTypedMedia("unavailable", retainedCount);
        diagnostics?.recordTypedMedia("failures", retainedCount);
        diagnostics?.recordUploadFailure("prepare", unavailableReason);
        const safePreview = `upload unavailable: ${unavailableReason}`;
        rewritePendingPlaceholder(attributes, item, {
          state: "failed",
          safe_preview: safePreview,
        });
        continue;
      }
      if (item.byteLength > Math.min(authority.maxPayloadBytes, DEFAULT_MAX_TYPED_MEDIA_BYTES)) {
        failed = true;
        diagnostics?.recordTypedMedia("failures", retainedCount);
        diagnostics?.recordUploadFailure("validate", "payload_too_large");
        const safePreview = "upload failed: validate/payload_too_large";
        rewritePendingPlaceholder(attributes, item, {
          state: "failed",
          safe_preview: safePreview,
        });
        continue;
      }
      if (!isSupportedTypedMediaMimeType(item.mimeType)) {
        failed = true;
        diagnostics?.recordTypedMedia("failures", retainedCount);
        diagnostics?.recordUploadFailure("validate", "unsupported_mime_type");
        const safePreview = "upload failed: validate/unsupported_mime_type";
        rewritePendingPlaceholder(attributes, item, {
          state: "failed",
          safe_preview: safePreview,
        });
        continue;
      }
      const payload: UploadPayload = {
        content: item.bytes,
        purpose: "typed_media",
        sha256: item.sha256,
        byteLength: item.byteLength,
        mimeType: item.mimeType,
        contentEncoding: "identity",
        idempotencyKey: `nl-ts-v1:typed_media:${item.sha256}:${item.mimeType}`,
        payloadSchema: "neatlogs.media.v1",
      };
      try {
        const receipt = assertReadyUploadReceipt(
          await uploadWithinDeadline(authority, payload, deadlineUnixMs),
          payload,
        );
        diagnostics?.recordTypedMedia("uploads", retainedCount);
        rewritePendingPlaceholder(attributes, item, {
          id: receipt.reference.id,
          reference: receipt.reference.id,
          source: "uploaded",
          sha256: receipt.reference.sha256,
          byte_length: receipt.reference.byteLength,
          mime_type: receipt.reference.mimeType,
          purpose: receipt.reference.purpose,
          state: "available",
          content_encoding: receipt.reference.contentEncoding,
        });
      } catch (error) {
        failed = true;
        const failure = safeFailure(error);
        diagnostics?.recordTypedMedia("failures", retainedCount);
        diagnostics?.recordUploadFailure(failure.stage, failure.reason);
        const safePreview = `upload failed: ${failure.stage}/${failure.reason}`;
        rewritePendingPlaceholder(attributes, item, {
          state: "failed",
          safe_preview: safePreview,
        });
      }
    }
  } finally {
    if (bucket) releasePendingAccount(bucket.account);
    const scrubbed = scrubUploadTokens(attributes);
    for (const key of Object.keys(attributes)) delete attributes[key];
    if (scrubbed && typeof scrubbed === "object" && !Array.isArray(scrubbed)) {
      Object.assign(attributes, scrubbed);
    }
  }
  return !failed;
}

/** Drop out-of-band bytes when a span is filtered or never exported. */
export function discardPendingMedia(span: object): void {
  const bucket = detachPendingBucket(span);
  if (bucket) releasePendingAccount(bucket.account);
}

/** Release all staged bytes owned by one SDK pipeline during shutdown. */
export function discardPendingMediaOwner(owner?: object): void {
  const state = pendingMediaOwners.get(owner ?? defaultPendingMediaOwner);
  if (!state) return;
  for (const account of [...state.accounts]) {
    const span = account.span.deref();
    if (span) {
      const bucket = pendingBySpan.get(span);
      if (bucket?.account === account) {
        pendingBySpan.delete(span);
        pendingFinalizer?.unregister(bucket.unregisterToken);
      }
    }
    releasePendingAccount(account);
  }
  pendingMediaOwners.delete(owner ?? defaultPendingMediaOwner);
}
