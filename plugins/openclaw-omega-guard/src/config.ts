import type { OmegaFailMode, OmegaPluginConfig, OmegaRuntimeMode } from "./types.js";

const DEFAULT_BASE_URL = "http://127.0.0.1:8080";
const DEFAULT_TENANT = "openclaw-default";

function asObject(value: unknown): Record<string, unknown> {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return {};
}

function asString(value: unknown, fallback = ""): string {
  if (typeof value !== "string") {
    return fallback;
  }
  const out = value.trim();
  return out || fallback;
}

function asBoolean(value: unknown, fallback: boolean): boolean {
  if (typeof value === "boolean") {
    return value;
  }
  return fallback;
}

function asInt(value: unknown, fallback: number): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.max(100, Math.trunc(value));
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number.parseInt(value, 10);
    if (Number.isFinite(parsed)) {
      return Math.max(100, parsed);
    }
  }
  return fallback;
}

function asRuntimeMode(value: unknown): OmegaRuntimeMode {
  const text = asString(value, "stateful").toLowerCase();
  return text === "stateless" ? "stateless" : "stateful";
}

function asFailMode(value: unknown): OmegaFailMode {
  const text = asString(value, "fail_closed").toLowerCase();
  return text === "fail_open" ? "fail_open" : "fail_closed";
}

function trimSlash(url: string): string {
  return url.replace(/\/+$/g, "");
}

export function loadPluginConfig(rawConfig: unknown): OmegaPluginConfig {
  const root = asObject(rawConfig);
  const omegaApiRaw = asObject(root.omegaApi);
  const omegaRaw = asObject(root.omega);
  const guardRaw = asObject(omegaRaw.guard);
  const alertsRaw = asObject(root.alerts);

  const cfg: OmegaPluginConfig = {
    tenantId: asString(root.tenantId, DEFAULT_TENANT),
    omegaApi: {
      baseUrl: trimSlash(asString(omegaApiRaw.baseUrl, DEFAULT_BASE_URL)),
      apiKey: asString(omegaApiRaw.apiKey, ""),
      hmacSecret: asString(omegaApiRaw.hmacSecret, ""),
      timeoutMs: asInt(omegaApiRaw.timeoutMs, 8000)
    },
    omega: {
      guard: {
        runtimeMode: asRuntimeMode(guardRaw.runtimeMode)
      },
      failMode: asFailMode(omegaRaw.failMode)
    },
    alerts: {
      enabled: asBoolean(alertsRaw.enabled, true)
    }
  };
  return cfg;
}
