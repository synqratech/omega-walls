# API Attachment Security Hardening (Proxy TLS + API Key/HMAC + JWS)

## Defaults

1. TLS is terminated on reverse proxy.
2. Uvicorn stays internal HTTP only.
3. `X-API-Key` and HMAC request signature are required.
4. Payloads are processed in-memory only and are never stored by default.
5. Attestation is optional and uses RS256 JWS.

## Reverse Proxy Requirements

1. Terminate TLS at proxy.
2. Forward requests to internal Uvicorn over private network.
3. Always set `X-Forwarded-Proto: https` (or `Forwarded: proto=https`).
4. Block direct internet traffic to Uvicorn port.

## Request Authentication

Required headers:

1. `X-API-Key`
2. `X-Signature`
3. `X-Timestamp` (unix seconds)
4. `X-Nonce`

Canonical string for HMAC:

1. `METHOD`
2. `PATH`
3. `sha256(body_bytes_hex)`
4. `tenant_id`
5. `request_id`
6. `X-Timestamp`
7. `X-Nonce`

Join with `\n`, then:

1. `sig = base64url(hmac_sha256(secret, canonical))`

## Attestation (Optional)

When `api.attestation.enabled=true`:

1. Response includes `attestation: { alg, kid, ts, jws }`.
2. `jws` is compact JWS signed with RS256.
3. Signed payload is full response body without `attestation`, plus `iat` and `exp`.

If signing key is missing/invalid:

1. Scan still returns `200`.
2. `attestation` is omitted.
3. reason code includes `attestation_unavailable`.

## Security Errors

1. `400 insecure_transport`
2. `401 unauthorized`
3. `401 invalid_signature`
4. `401 stale_timestamp`
5. `409 replay_detected`

## Evidence Contract (No Sensitive Text)

1. `evidence_id` is UUID (`uuid4`).
2. Response includes compact `evidence`:
   - `walls_triggered`
   - `rule_ids`
   - `chunk_ids`
   - `top_chunk_ids`
   - `text_included=false`
3. No raw content or snippets are returned in `evidence`/`policy_trace`.

## Debug Endpoint (Pilot FP/FN)

1. `POST /v1/scan/attachment/document_scan_report`
2. Purpose: return per-chunk evidence for FP/FN analysis in pilot.
3. Requires same auth/TLS controls as main endpoint.
4. Enabled only with `api.debug.enable_document_scan_report=true`.
5. Payload includes only structured signals:
   - chunk ids
   - wall scores/active walls
   - pattern/rule ids
   - aggregate chunk-pipeline scores
6. Raw text/snippets are not returned (`text_included=false`).

## Logging Policy

Audit logs are structured and contain only:

1. hashed tenant id
2. request metadata (`request_id`, mime, extension, payload size)
3. verdict/risk/reasons/evidence id
4. compact policy trace and pattern IDs

Never logged:

1. raw `extracted_text`
2. raw `file_base64`
3. raw chunk text
