# Shared Responsibility (Self-Hosted First)

Document version: v0.1  
Status: Draft  
Last updated: 2026-05-13  
Owner: Founder / Product Security  
Legal status: Operational guidance only.

## 1. Vendor Responsibility

Vendor provides product capabilities and baseline security engineering for:

- runtime control logic
- policy execution paths
- tool-gateway interception model
- audit/event semantics
- documented configuration behavior

## 2. Customer Responsibility

Customer remains responsible for deployment and operation of:

- infrastructure hardening and network controls
- identity/access management
- secret storage and rotation
- logging retention policies in self-hosted systems
- correct integration so all relevant traffic passes through guard boundaries

## 3. Joint Responsibility Areas

- incident handling coordination
- tuning for false positives/false negatives
- safe rollout from monitor to enforce modes
- governance and change-control in production environments
