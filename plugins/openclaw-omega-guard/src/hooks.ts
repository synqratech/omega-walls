import { mapOmegaDecision, toHookDecision } from "./decision.js";
import { extractIdentity, extractTextPayload } from "./identity.js";
import { redactForAlert } from "./redaction.js";
import type { OmegaApiClient } from "./omega-client.js";
import type { OmegaPluginConfig } from "./types.js";

function emitAlert(api: any, level: "warn" | "error", message: string, payload: Record<string, unknown>): void {
  if (typeof api?.log === "function") {
    api.log(level, message, payload);
    return;
  }
  if (api?.logger && typeof api.logger[level] === "function") {
    api.logger[level](message, payload);
  }
}

export function registerOmegaHooks(api: any, cfg: OmegaPluginConfig, client: OmegaApiClient): void {
  const hookHandler = async (hookName: string, ctx: unknown): Promise<Record<string, unknown> | undefined> => {
    const identity = extractIdentity(ctx, {
      tenantId: cfg.tenantId,
      sessionFallback: `openclaw-${hookName}`
    });
    const extractedText = extractTextPayload(ctx, 8000);
    if (!extractedText) {
      return undefined;
    }
    try {
      const response = await client.scanExtractedText({
        identity,
        runtimeMode: cfg.omega.guard.runtimeMode,
        extractedText
      });
      const decision = mapOmegaDecision(response);
      if (cfg.alerts.enabled && decision.kind !== "allow") {
        emitAlert(api, "warn", `omega_guard_${decision.kind}`, {
          hook: hookName,
          session_id: identity.sessionId,
          actor_id: identity.actorId,
          trace_id: decision.traceId,
          decision_id: decision.decisionId,
          payload: redactForAlert(response)
        });
      }
      return toHookDecision(decision);
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      emitAlert(api, "error", "omega_guard_error", {
        hook: hookName,
        session_id: identity.sessionId,
        detail
      });
      if (cfg.omega.failMode === "fail_open") {
        return undefined;
      }
      return {
        block: true,
        reason: "omega_guard_unavailable"
      };
    }
  };

  const registerOne = (hookName: string): void => {
    try {
      if (typeof api?.on === "function") {
        api.on(hookName, async (ctx: unknown) => hookHandler(hookName, ctx));
      }
    } catch (err) {
      emitAlert(api, "warn", "omega_hook_registration_failed", {
        hook: hookName,
        detail: err instanceof Error ? err.message : String(err)
      });
    }
  };

  registerOne("before_agent_reply");
  registerOne("before_tool_call");
  registerOne("message_sending");

  const registerResetHook = (hookName: string): void => {
    try {
      if (typeof api?.on !== "function") {
        return;
      }
      api.on(hookName, async (ctx: unknown) => {
        const identity = extractIdentity(ctx, {
          tenantId: cfg.tenantId,
          sessionFallback: `openclaw-${hookName}`
        });
        try {
          await client.resetSession({
            tenantId: identity.tenantId,
            sessionId: identity.sessionId,
            actorId: identity.actorId
          });
        } catch (err) {
          emitAlert(api, "warn", "omega_session_reset_failed", {
            hook: hookName,
            session_id: identity.sessionId,
            detail: err instanceof Error ? err.message : String(err)
          });
        }
        return undefined;
      });
    } catch (err) {
      emitAlert(api, "warn", "omega_reset_hook_registration_failed", {
        hook: hookName,
        detail: err instanceof Error ? err.message : String(err)
      });
    }
  };

  registerResetHook("conversation_end");
  registerResetHook("session_end");
}
