import test from "node:test";
import assert from "node:assert/strict";

import { mapOmegaDecision, toHookDecision } from "../src/decision.js";

test("mapOmegaDecision block-like outcomes -> block", () => {
  const decision = mapOmegaDecision({
    control_outcome: "SOFT_BLOCK",
    reasons: ["tool_abuse"],
    policy_id: "policy.tool_abuse",
    fallback_hint: "review_tool_policy",
    incident_artifact_id: "ia_123"
  });
  assert.equal(decision.kind, "block");
  assert.equal(decision.reason, "tool_abuse");
  const hook = toHookDecision(decision);
  assert.equal(hook?.block, true);
  assert.equal(hook?.reason, "tool_abuse");
  assert.equal(hook?.action, "SOFT_BLOCK");
  assert.equal(hook?.controlOutcome, "SOFT_BLOCK");
  assert.equal(hook?.incidentArtifactId, "ia_123");
  assert.equal(hook?.policyId, "policy.tool_abuse");
  assert.equal(hook?.fallbackHint, "review_tool_policy");
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
  assert.equal(hook?.action, "ALLOW");
  assert.equal(hook?.controlOutcome, "ALLOW");
});
