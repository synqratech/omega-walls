import type { OmegaScanResponse } from "./types.js";

export function redactForAlert(payload: OmegaScanResponse): Record<string, unknown> {
  const policyTrace = (payload.policy_trace ?? {}) as Record<string, unknown>;
  return {
    control_outcome: String(payload.control_outcome ?? ""),
    reasons: Array.isArray(payload.reasons) ? payload.reasons : [],
    action_types: Array.isArray(policyTrace.action_types) ? policyTrace.action_types : [],
    trace_id: String(policyTrace.trace_id ?? ""),
    decision_id: String(policyTrace.decision_id ?? ""),
    incident_artifact_id: String(payload.incident_artifact_id ?? ""),
    risk_score: typeof payload.risk_score === "number" ? payload.risk_score : null
  };
}
