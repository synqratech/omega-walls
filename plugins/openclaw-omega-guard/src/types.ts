export type OmegaRuntimeMode = "stateful" | "stateless";
export type OmegaFailMode = "fail_closed" | "fail_open";

export interface OpenClawIdentity {
  tenantId: string;
  sessionId: string;
  actorId: string;
}

export interface OmegaApiConfig {
  baseUrl: string;
  apiKey: string;
  hmacSecret: string;
  timeoutMs: number;
}

export interface OmegaPluginConfig {
  tenantId: string;
  omegaApi: OmegaApiConfig;
  omega: {
    guard: {
      runtimeMode: OmegaRuntimeMode;
    };
    failMode: OmegaFailMode;
  };
  alerts: {
    enabled: boolean;
  };
}

export interface OmegaScanRequest {
  identity: OpenClawIdentity;
  extractedText: string;
  runtimeMode: OmegaRuntimeMode;
  sourceType?: string;
  filename?: string;
  mime?: string;
}

export interface OmegaScanResponse {
  request_id?: string;
  control_outcome?: string;
  reasons?: string[];
  policy_trace?: {
    action_types?: string[];
    trace_id?: string;
    decision_id?: string;
  };
  approval_required?: boolean;
  approval_id?: string;
  approval_status?: string;
  incident_artifact_id?: string;
  risk_score?: number;
  monitor?: {
    false_positive_hint?: string;
  };
  [key: string]: unknown;
}

export interface GuardDecision {
  kind: "allow" | "block" | "require_approval";
  reason?: string;
  traceId?: string;
  decisionId?: string;
  incidentArtifactId?: string;
  controlOutcome?: string;
}
