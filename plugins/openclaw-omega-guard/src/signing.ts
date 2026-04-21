import { createHash, createHmac, randomUUID } from "node:crypto";

export interface CanonicalRequestParts {
  method: string;
  path: string;
  bodySha256Hex: string;
  tenantId: string;
  requestId: string;
  timestamp: string;
  nonce: string;
}

export function sha256Hex(input: string | Buffer): string {
  return createHash("sha256").update(input).digest("hex");
}

export function b64Url(input: Buffer): string {
  return input.toString("base64url");
}

export function canonicalRequestString(parts: CanonicalRequestParts): string {
  return [
    parts.method.toUpperCase().trim(),
    parts.path.trim(),
    parts.bodySha256Hex.trim(),
    parts.tenantId.trim(),
    parts.requestId.trim(),
    parts.timestamp.trim(),
    parts.nonce.trim()
  ].join("\n");
}

export function signCanonicalString(canonical: string, secret: string): string {
  const digest = createHmac("sha256", secret).update(canonical).digest();
  return b64Url(digest);
}

export interface SignedHeadersInput {
  method: string;
  path: string;
  bodyBytes: Buffer;
  tenantId: string;
  requestId: string;
  hmacSecret: string;
}

export function buildSignedHeaders(input: SignedHeadersInput): {
  timestamp: string;
  nonce: string;
  signature: string;
  bodyHash: string;
} {
  const timestamp = String(Math.floor(Date.now() / 1000));
  const nonce = randomUUID().replace(/-/g, "");
  const bodyHash = sha256Hex(input.bodyBytes);
  const canonical = canonicalRequestString({
    method: input.method,
    path: input.path,
    bodySha256Hex: bodyHash,
    tenantId: input.tenantId,
    requestId: input.requestId,
    timestamp,
    nonce
  });
  const signature = signCanonicalString(canonical, input.hmacSecret);
  return { timestamp, nonce, signature, bodyHash };
}
