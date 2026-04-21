import type { OpenClawIdentity } from "./types.js";

const SESSION_KEYS = [
  "session_id",
  "sessionId",
  "conversation_id",
  "conversationId",
  "thread_id",
  "threadId",
  "run_id",
  "runId"
];

const ACTOR_KEYS = [
  "actor_id",
  "actorId",
  "user_id",
  "userId",
  "sender_id",
  "senderId",
  "principal_id",
  "principalId"
];

function asObject(value: unknown): Record<string, unknown> | null {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return null;
}

function deepLookup(input: unknown, keys: string[], depth = 0): string | null {
  if (depth > 4) {
    return null;
  }
  const obj = asObject(input);
  if (!obj) {
    return null;
  }

  for (const key of keys) {
    const value = obj[key];
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }

  for (const nestedKey of ["context", "metadata", "state", "request", "message", "tool", "runtime"]) {
    const nested = obj[nestedKey];
    const found = deepLookup(nested, keys, depth + 1);
    if (found) {
      return found;
    }
  }

  return null;
}

function compact(text: unknown): string {
  if (typeof text !== "string") {
    return "";
  }
  return text.trim();
}

export function extractIdentity(
  source: unknown,
  defaults: { tenantId: string; sessionFallback?: string } = { tenantId: "openclaw-default" }
): OpenClawIdentity {
  const sessionId = deepLookup(source, SESSION_KEYS) || compact(defaults.sessionFallback) || "openclaw-session-default";
  const actorId = deepLookup(source, ACTOR_KEYS) || sessionId;
  return {
    tenantId: compact(defaults.tenantId) || "openclaw-default",
    sessionId,
    actorId
  };
}

function normalizeContent(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  if (Array.isArray(value)) {
    return value.map((x) => normalizeContent(x)).filter(Boolean).join("\n");
  }
  const obj = asObject(value);
  if (!obj) {
    return "";
  }
  if (typeof obj.text === "string" && obj.text.trim()) {
    return obj.text;
  }
  if (typeof obj.content === "string" && obj.content.trim()) {
    return obj.content;
  }
  return "";
}

export function extractTextPayload(source: unknown, maxChars = 8000): string {
  const obj = asObject(source);
  if (!obj) {
    return "";
  }
  const candidates: unknown[] = [
    obj.extractedText,
    obj.inputText,
    obj.userText,
    obj.prompt,
    obj.text,
    obj.content,
    obj.message,
    obj.messages,
    obj.toolInput,
    obj.args
  ];
  for (const candidate of candidates) {
    const normalized = normalizeContent(candidate).trim();
    if (normalized) {
      return normalized.slice(0, maxChars);
    }
  }
  return "";
}
