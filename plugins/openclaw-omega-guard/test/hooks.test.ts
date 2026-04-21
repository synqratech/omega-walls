import test from "node:test";
import assert from "node:assert/strict";

import { registerOmegaHooks } from "../src/hooks.js";
import { loadPluginConfig } from "../src/config.js";

test("registerOmegaHooks wires before_tool_call and returns block on fail_closed error", async () => {
  const hooks = new Map<string, (ctx: unknown) => Promise<Record<string, unknown> | undefined>>();
  const api = {
    on(name: string, handler: (ctx: unknown) => Promise<Record<string, unknown> | undefined>) {
      hooks.set(name, handler);
    }
  };
  const cfg = loadPluginConfig({
    tenantId: "tenant-x",
    omegaApi: { baseUrl: "http://127.0.0.1:8080", apiKey: "k", hmacSecret: "s" },
    omega: { failMode: "fail_closed", guard: { runtimeMode: "stateful" } }
  });
  const client = {
    async scanExtractedText() {
      throw new Error("boom");
    },
    async resetSession() {
      return { reset: true };
    }
  };

  registerOmegaHooks(api, cfg, client as any);
  assert.ok(hooks.has("before_tool_call"));
  assert.ok(hooks.has("conversation_end"));
  assert.ok(hooks.has("session_end"));
  const decision = await hooks.get("before_tool_call")?.({ toolInput: "x", session_id: "s-1" });
  assert.equal(decision?.block, true);
});
