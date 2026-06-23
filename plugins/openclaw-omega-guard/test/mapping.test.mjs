import test from "node:test";
import assert from "node:assert/strict";
import { failClosedDecision, mapOmegaDecision } from "../dist/hooks.js";

test("block verdict maps to OpenClaw block", () => {
  assert.deepEqual(mapOmegaDecision({ verdict: "block", reasons: ["reason_spike"] }), {
    block: true,
    reason: "reason_spike",
  });
});

test("quarantine maps to approval", () => {
  assert.equal(mapOmegaDecision({ verdict: "quarantine" })?.requireApproval, true);
});

test("guard failure is fail closed", () => {
  assert.equal(failClosedDecision().block, true);
});
