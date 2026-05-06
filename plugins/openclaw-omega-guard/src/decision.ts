import type { GuardDecision, OmegaScanResponse } from "./types.js";

const BLOCK_OUTCOMES = new Set([
  "OFF",
  "BLOCK",
  "SOFT_BLOCK",
  "TOOL_FREEZE",
  "SOURCE_QUARANTINE",
  "QUARANTINE",
  "WARN"
]);

const ESCALATE_OUTCOMES = new Set(["HUMAN_ESCALATE", "REQUIRE_APPROVAL", "ESCALATE"]);

function pickReason(payload: OmegaScanResponse): string | undefined {
  const reasons = Array.isArray(payload.reasons) ? payload.reasons : [];
  if (reasons.length > 0) {
    return String(reasons[0]);
  }
  const hint = payload.monitor?.false_positive_hint;
  if (typeof hint === "string" && hint.trim()) {
    return hint.trim();
  }
  return undefined;
}

export function mapOmegaDecision(payload: OmegaScanResponse): GuardDecision {
  const controlOutcome = String(payload.control_outcome ?? "ALLOW").toUpperCase();
  const policyTrace = (payload.policy_trace ?? {}) as Record<string, unknown>;
  const traceId = typeof policyTrace.trace_id === "string" ? policyTrace.trace_id : undefined;
  const decisionId = typeof policyTrace.decision_id === "string" ? policyTrace.decision_id : undefined;
  const incidentArtifactId =
    typeof payload.incident_artifact_id === "string" ? payload.incident_artifact_id : undefined;
  const policyId =
    typeof payload.policy_id === "string"
      ? payload.policy_id
      : (typeof (policyTrace as Record<string, unknown>).policy_id === "string"
        ? String((policyTrace as Record<string, unknown>).policy_id)
        : undefined);
  const fallbackHint =
    typeof payload.fallback_hint === "string" ? payload.fallback_hint : undefined;
  const reason = pickReason(payload);

  if (Boolean(payload.approval_required) || ESCALATE_OUTCOMES.has(controlOutcome)) {
    return {
      kind: "require_approval",
      reason,
      traceId,
      decisionId,
      incidentArtifactId,
      policyId,
      fallbackHint,
      controlOutcome
    };
  }

  if (BLOCK_OUTCOMES.has(controlOutcome)) {
    return {
      kind: "block",
      reason,
      traceId,
      decisionId,
      incidentArtifactId,
      policyId,
      fallbackHint,
      controlOutcome
    };
  }

  return {
    kind: "allow",
    reason,
    traceId,
    decisionId,
    incidentArtifactId,
    policyId,
    fallbackHint,
    controlOutcome
  };
}

export function toHookDecision(decision: GuardDecision): Record<string, unknown> | undefined {
  if (decision.kind === "allow") {
    return undefined;
  }
  if (decision.kind === "block") {
    return {
      block: true,
      reason: decision.reason ?? "omega_blocked",
      action: decision.controlOutcome ?? "BLOCK",
      controlOutcome: decision.controlOutcome ?? "BLOCK",
      incidentArtifactId: decision.incidentArtifactId,
      policyId: decision.policyId,
      fallbackHint: decision.fallbackHint,
      traceId: decision.traceId,
      decisionId: decision.decisionId
    };
  }
  return {
    requireApproval: true,
    reason: decision.reason ?? "omega_requires_approval",
    action: decision.controlOutcome ?? "REQUIRE_APPROVAL",
    controlOutcome: decision.controlOutcome ?? "REQUIRE_APPROVAL",
    incidentArtifactId: decision.incidentArtifactId,
    policyId: decision.policyId,
    fallbackHint: decision.fallbackHint,
    traceId: decision.traceId,
    decisionId: decision.decisionId
  };
}
