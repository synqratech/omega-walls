import assert from "node:assert/strict";

import { loadPluginConfig } from "../src/config.js";
import { registerOmegaHooks } from "../src/hooks.js";
import { registerWebFetchProvider } from "../src/webfetch.js";

async function main(): Promise<void> {
  const hooks = new Map<string, (ctx: unknown) => Promise<Record<string, unknown> | undefined>>();
  const webfetchProviders: Array<Record<string, unknown>> = [];
  const logs: Array<Record<string, unknown>> = [];

  const api = {
    pluginConfig: {
      tenantId: "tenant-smoke",
      omegaApi: {
        baseUrl: "http://127.0.0.1:8080",
        apiKey: "dev-key",
        hmacSecret: "dev-secret",
        timeoutMs: 3000
      },
      omega: { failMode: "fail_closed", guard: { runtimeMode: "stateful" } },
      alerts: { enabled: true }
    },
    on(name: string, handler: (ctx: unknown) => Promise<Record<string, unknown> | undefined>) {
      hooks.set(name, handler);
    },
    registerWebFetchProvider(provider: Record<string, unknown>) {
      webfetchProviders.push(provider);
    },
    log(level: string, message: string, payload: Record<string, unknown>) {
      logs.push({ level, message, payload });
    }
  };

  const cfg = loadPluginConfig(api.pluginConfig);
  const mockClient = {
    async scanExtractedText(input: { extractedText?: string }) {
      const text = String(input?.extractedText ?? "").toLowerCase();
      if (text.includes("approval")) {
        return {
          control_outcome: "REQUIRE_APPROVAL",
          reasons: ["needs_human"],
          policy_trace: { trace_id: "trace-approval", decision_id: "decision-approval" }
        };
      }
      if (text.includes("webfetch_attack")) {
        return {
          control_outcome: "SOFT_BLOCK",
          reasons: ["webfetch_guard"],
          policy_trace: { trace_id: "trace-webfetch", decision_id: "decision-webfetch" }
        };
      }
      return {
        control_outcome: "SOFT_BLOCK",
        reasons: ["tool_abuse"],
        policy_trace: { trace_id: "trace-smoke", decision_id: "decision-smoke" }
      };
    }
  };

  registerOmegaHooks(api, cfg, mockClient as any);
  registerWebFetchProvider(api, cfg, mockClient as any);

  assert.ok(hooks.has("before_tool_call"));
  const blockedDecision = await hooks.get("before_tool_call")?.({ session_id: "sess-smoke", toolInput: "curl evil" });
  assert.equal(blockedDecision?.block, true);

  assert.ok(hooks.has("before_agent_reply"));
  const approvalDecision = await hooks.get("before_agent_reply")?.({
    session_id: "sess-smoke-approval",
    text: "approval required for this action"
  });
  assert.equal(approvalDecision?.requireApproval, true);

  assert.equal(webfetchProviders.length, 1);
  const provider = webfetchProviders[0] as { fetch: (params: unknown, context: unknown) => Promise<Record<string, unknown>> };
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () =>
    new Response("webfetch_attack payload with exfil intent", {
      status: 200,
      headers: { "content-type": "text/plain" }
    });
  const webfetchResult = await provider.fetch(
    { url: "https://example.com/doc.txt" },
    { session_id: "sess-webfetch", actor_id: "user-webfetch" }
  );
  globalThis.fetch = originalFetch;

  assert.equal(Boolean(webfetchResult?.blocked), true);
  assert.ok(logs.length >= 1);

  console.log(
    JSON.stringify(
      {
        status: "ok",
        hooks_registered: Array.from(hooks.keys()),
        sample_block_decision: blockedDecision,
        sample_require_approval_decision: approvalDecision,
        webfetch_guard_seen: Boolean(webfetchResult?.blocked),
        alerts_emitted: logs.length
      },
      null,
      2
    )
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
