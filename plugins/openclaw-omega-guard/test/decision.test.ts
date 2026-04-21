import test from "node:test";
import assert from "node:assert/strict";

import { mapOmegaDecision, toHookDecision } from "../src/decision.js";

test("mapOmegaDecision block-like outcomes -> block", () => {
  const decision = mapOmegaDecision({
    control_outcome: "SOFT_BLOCK",
    reasons: ["tool_abuse"]
  });
  assert.equal(decision.kind, "block");
  assert.equal(decision.reason, "tool_abuse");
  assert.deepEqual(toHookDecision(decision), {
    block: true,
    reason: "tool_abuse",
    traceId: undefined,
    decisionId: undefined
  });
});

test("mapOmegaDecision approval_required -> require_approval", () => {
  const decision = mapOmegaDecision({
    control_outcome: "ALLOW",
    approval_required: true,
    reasons: ["human_escalate"]
  });
  assert.equal(decision.kind, "require_approval");
  const hook = toHookDecision(decision);
  assert.equal(hook?.requireApproval, true);
});
