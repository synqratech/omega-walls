# Telemetry and Data Handling

Document version: v0.1  
Status: Draft  
Last updated: 2026-05-13  
Owner: Founder / Product Security  
Legal status: Operational baseline, subject to customer configuration and contract.

## 1. Telemetry Baseline

Telemetry should be explicitly configurable and expected to be opt-in for enterprise-sensitive environments.

## 2. Allowed Telemetry Examples

- product/version metadata
- deployment mode
- anonymized instance identifier
- event category counters
- non-content diagnostic status

## 3. Prohibited Telemetry Examples

The following should not be exported by default:

- raw prompts or documents
- raw retrieval chunks or memory content
- secrets, keys, tokens, passwords
- direct personal identifiers
- IP addresses and hostnames
- full sensitive stack traces and internal path disclosures

## 4. Opt-Out and Controls

Telemetry behavior is controlled via configuration/environment settings.
Disabling telemetry should stop further outbound transmission and clear pending telemetry queues where implemented.

## 5. Retention Baseline

Vendor-held telemetry (if enabled) follows limited retention with preference for aggregated signals over raw events.
Customer self-hosted logs remain customer-controlled.
