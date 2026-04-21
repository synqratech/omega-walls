import { buildSignedHeaders } from "./signing.js";
import type { OmegaPluginConfig, OmegaScanRequest, OmegaScanResponse } from "./types.js";

export class OmegaApiClient {
  private readonly cfg: OmegaPluginConfig;
  private readonly fetchImpl: typeof fetch;

  constructor(cfg: OmegaPluginConfig, fetchImpl: typeof fetch = fetch) {
    this.cfg = cfg;
    this.fetchImpl = fetchImpl;
  }

  private async postJson(path: string, body: Record<string, unknown>): Promise<OmegaScanResponse> {
    const api = this.cfg.omegaApi;
    if (!api.apiKey.trim()) {
      throw new Error("omega_api_key_missing");
    }
    if (!api.hmacSecret.trim()) {
      throw new Error("omega_hmac_secret_missing");
    }
    const bodyBytes = Buffer.from(JSON.stringify(body), "utf-8");
    const requestId = String(body.request_id ?? "");
    const signed = buildSignedHeaders({
      method: "POST",
      path,
      bodyBytes,
      tenantId: String(body.tenant_id ?? ""),
      requestId,
      hmacSecret: api.hmacSecret
    });

    const res = await this.fetchImpl(`${api.baseUrl}${path}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": api.apiKey,
        "X-Timestamp": signed.timestamp,
        "X-Nonce": signed.nonce,
        "X-Signature": signed.signature
      },
      body: bodyBytes,
      signal: AbortSignal.timeout(api.timeoutMs)
    });
    if (!res.ok) {
      const detail = await res.text();
      throw new Error(`omega_api_error_${res.status}:${detail.slice(0, 300)}`);
    }
    const payload = (await res.json()) as OmegaScanResponse;
    return payload;
  }

  async scanExtractedText(input: OmegaScanRequest): Promise<OmegaScanResponse> {
    const requestId = `${Date.now()}-${Math.random().toString(16).slice(2, 10)}`;
    return this.postJson("/v1/scan/attachment", {
      tenant_id: input.identity.tenantId,
      request_id: requestId,
      session_id: input.identity.sessionId,
      actor_id: input.identity.actorId,
      runtime_mode: input.runtimeMode,
      extracted_text: input.extractedText,
      filename: input.filename ?? "openclaw.txt",
      mime: input.mime ?? "text/plain",
      source_type: input.sourceType ?? "web"
    });
  }

  async resetSession(input: {
    tenantId: string;
    sessionId: string;
    actorId?: string;
  }): Promise<OmegaScanResponse> {
    const requestId = `${Date.now()}-${Math.random().toString(16).slice(2, 10)}`;
    return this.postJson("/v1/session/reset", {
      tenant_id: input.tenantId,
      request_id: requestId,
      session_id: input.sessionId,
      actor_id: input.actorId ?? input.sessionId
    });
  }
}
