import test from "node:test";
import assert from "node:assert/strict";

import { canonicalRequestString, sha256Hex, signCanonicalString } from "../src/signing.js";

test("canonicalRequestString is deterministic", () => {
  const value = canonicalRequestString({
    method: "post",
    path: "/v1/scan/attachment",
    bodySha256Hex: sha256Hex('{"x":1}'),
    tenantId: "tenant-a",
    requestId: "req-1",
    timestamp: "123",
    nonce: "n-1"
  });
  assert.equal(
    value,
    `POST
/v1/scan/attachment
${sha256Hex('{"x":1}')}
tenant-a
req-1
123
n-1`
  );
});

test("signCanonicalString returns stable base64url output", () => {
  const signed = signCanonicalString("A\nB\nC", "secret-1");
  const signed2 = signCanonicalString("A\nB\nC", "secret-1");
  assert.equal(signed, signed2);
  assert.ok(signed.length > 10);
});
