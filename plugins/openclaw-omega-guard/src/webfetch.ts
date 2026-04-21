import { mapOmegaDecision } from "./decision.js";
import { extractIdentity } from "./identity.js";
import type { OmegaApiClient } from "./omega-client.js";
import type { OmegaPluginConfig } from "./types.js";

function asUrl(input: unknown): string {
  if (typeof input === "string" && input.trim()) {
    return input.trim();
  }
  if (input && typeof input === "object" && "url" in (input as Record<string, unknown>)) {
    const url = (input as Record<string, unknown>).url;
    if (typeof url === "string" && url.trim()) {
      return url.trim();
    }
  }
  return "";
}

export function registerWebFetchProvider(api: any, cfg: OmegaPluginConfig, client: OmegaApiClient): void {
  if (typeof api?.registerWebFetchProvider !== "function") {
    return;
  }

  api.registerWebFetchProvider({
    id: "omega-guarded-fetch",
    name: "Omega Guarded Fetch",
    description: "Web fetch provider with Omega pre-ingestion guard checks",
    async fetch(params: unknown, context: unknown) {
      const url = asUrl(params);
      if (!url) {
        return { ok: false, error: "missing_url" };
      }

      let responseText = "";
      let status = 0;
      let contentType = "text/plain";
      try {
        const res = await fetch(url, { method: "GET", signal: AbortSignal.timeout(cfg.omegaApi.timeoutMs) });
        status = res.status;
        contentType = res.headers.get("content-type") ?? "text/plain";
        responseText = await res.text();
      } catch (err) {
        return {
          ok: false,
          error: "upstream_fetch_failed",
          detail: err instanceof Error ? err.message : String(err)
        };
      }

      const identity = extractIdentity(context, {
        tenantId: cfg.tenantId,
        sessionFallback: `webfetch:${url}`
      });
      try {
        const scan = await client.scanExtractedText({
          identity,
          runtimeMode: cfg.omega.guard.runtimeMode,
          extractedText: responseText,
          sourceType: "web",
          filename: "web_fetch.txt",
          mime: contentType
        });
        const decision = mapOmegaDecision(scan);
        if (decision.kind === "allow") {
          return {
            ok: true,
            url,
            status,
            contentType,
            content: responseText
          };
        }
        if (decision.kind === "require_approval") {
          return {
            ok: false,
            requireApproval: true,
            reason: decision.reason ?? "omega_requires_approval",
            traceId: decision.traceId,
            decisionId: decision.decisionId
          };
        }
        return {
          ok: false,
          blocked: true,
          reason: decision.reason ?? "omega_blocked",
          traceId: decision.traceId,
          decisionId: decision.decisionId,
          content: "[REDACTED_BY_OMEGA]"
        };
      } catch (err) {
        if (cfg.omega.failMode === "fail_open") {
          return {
            ok: true,
            url,
            status,
            contentType,
            content: responseText,
            warning: "omega_guard_unavailable_fail_open"
          };
        }
        return {
          ok: false,
          blocked: true,
          reason: "omega_guard_unavailable",
          detail: err instanceof Error ? err.message : String(err)
        };
      }
    }
  });
}
