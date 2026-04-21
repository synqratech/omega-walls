import test from "node:test";
import assert from "node:assert/strict";

import { extractIdentity, extractTextPayload } from "../src/identity.js";

test("extractIdentity maps nested session/actor ids deterministically", () => {
  const payload = {
    context: {
      metadata: {
        session_id: "sess-42",
        user_id: "user-9"
      }
    }
  };
  const identity = extractIdentity(payload, { tenantId: "tenant-1" });
  assert.equal(identity.tenantId, "tenant-1");
  assert.equal(identity.sessionId, "sess-42");
  assert.equal(identity.actorId, "user-9");
});

test("extractTextPayload handles message arrays", () => {
  const payload = {
    messages: [{ content: "hello" }, { text: "world" }]
  };
  const text = extractTextPayload(payload, 1024);
  assert.equal(text, "hello\nworld");
});
