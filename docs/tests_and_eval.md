# Omega Walls: Tests & Evaluation (v1)

This document defines:
- which datasets are treated as **ground truth**,
- what metrics we track,
- CI gates (what fails the build),
- and the evaluation workflow to keep tuning reproducible.

---

## 1) Ground-truth datasets (JSONL)

### 1.1 Canonical test sets (must exist in repo)

```
tests/
  data/session_benchmark/agentdojo_cocktail_mini_smoke_v1.jsonl
scripts/
  quick_demo.py
  eval_agentdojo_stateful_mini.py
```

**Truth rule (Lean OSS):** the mini session pack above is the canonical quickstart smoke fixture.

### 1.2 JSONL schema (v1)
Each line:

```json
{
  "id": "POS-001",
  "family": "single|cocktail|obfuscation|paraphrase|distributed|hard_negative",
  "targets": ["override_instructions"],
  "text": "...",
  "expected_nonzero": ["override_instructions"],
  "expected_all_zero": false,
  "note": "why this exists"
}
```

**Interpretation**
- If `expected_all_zero == true`: we assert `sum(v)==0` (after polarity gating).
- Else: for each wall in `expected_nonzero` assert `v[wall] > 0`.

---

## 2) Unit-level tests

These tests isolate regressions to one component.

### 2.1 Projector tests (π₀)

**Inputs:** JSONL datasets.

**Assertions**
1) **Hard negatives**: for every item
   - `sum(v) == 0`
   - `max(polarity) <= 0`

2) **Positives (pos_20 + obf_20)**:
   - `v[w] > 0` for all expected walls `w`
   - recommended: `polarity[w] == +1` for expected walls

**Metrics logged (per wall)**
- `hit_rate[k] = hits[k] / total_targeted[k]`
- `false_positive_rate = FP / N_hard_neg`

**CI gates (π₀)**
- Hard negatives: **0 false positives** allowed (strict v1)
- Positives: `hit_rate >= 0.95` overall AND `>= 0.90` per wall

> If you relax hard negatives later, do it explicitly with a version bump and FP budget.

---

### 2.2 Ω-core tests

Use synthetic packets with known `v_{t,j}` to validate math:

**Assertions**
- `ε-floor` behavior: values < ε become 0
- aggregation uses sum
- toxicity `p = 1-exp(-αv)` within tolerance
- synergy `p += β C_S(p)` within tolerance
- update `m_{t+1} = λ m_t + v ⊙ p`
- `Off` reasons match thresholds exactly
- attribution uses `e_{t,j}=v_{t,j} ⊙ p`, `c=sum(e)` and γ-rule top docs

**CI gates (Ω-core)**
- all deterministic tests pass
- numeric tolerances stable (`1e-6` recommended)

---

### 2.3 OffPolicy tests

Given crafted `OmegaStepResult`, ensure:
- `off=True` → decision contains at least 1 action
- `SOFT_BLOCK` contains `top_docs`
- if tool wall participates → includes `TOOL_FREEZE`
- if exfil wall participates → includes `HUMAN_ESCALATE`
- actions are deterministic for identical inputs

**CI gates**
- all action mapping invariants pass

---

### 2.4 ToolGateway tests

Given active actions, ensure:
- if `TOOLS_DISABLED` active → tool denied (fail closed)
- allowlist mode denies unknown tools
- logs always emitted

---

## 3) Integration-level tests (end-to-end)

These tests validate that Ω is positioned correctly in the RAG/agent loop.

### 3.1 RAG pipeline integration checklist test
Create a minimal harness that:
1) Simulates retrieval results as a packet `X_t`
2) Runs π + Ω
3) Ensures blocked docs are excluded from context builder
4) Ensures tools can only be executed via ToolGateway
5) Ensures `omega_off_v1` event is emitted on Off

**CI gate**
- must pass with a small deterministic fixture.

---

## 4) Session-pack evaluation (Lean OSS)

### 4.1 Pack source
The Lean OSS quickstart uses `tests/data/session_benchmark/agentdojo_cocktail_mini_smoke_v1.jsonl`.
Advanced users can build a mini pack from local AgentDojo runs using `scripts/build_agentdojo_cocktail_mini_pack.py`.
- `label="malicious"`
- `targets` walls
- `family` (obfuscation/paraphrase/cocktail/distributed)

### 4.2 Metrics on generator corpus

Let \(D\) be generated dataset size.

#### A) Hit-rate by wall
For each wall k:
- `targeted_k`: number of samples that include wall k in `targets`
- `hits_k`: number of those where `v[k] > 0`

\[
\mathrm{hit\_rate}_k = rac{\mathrm{hits}_k}{\mathrm{targeted}_k}.
\]

#### B) Cocktail coverage
For samples targeting >=2 walls:
- `multi_hit = count_nonzero(v) >= 2`

\[
\mathrm{multi\_hit\_rate} = rac{\#\{x: |targets(x)|\ge 2 \land \#\{k:v_k(x)>0\}\ge 2\}}{\#\{x:|targets(x)|\ge 2\}}.
\]

#### C) Obfuscation robustness
For each obfuscation operator `op`:
- hit-rate restricted to samples where `op` was applied.

#### D) Adaptive-evasion score (optional but recommended)
If your runner searches for a variant that minimizes `sum(v)`:
- track `min_sum_v` distribution and top bypass examples.

### 4.3 CI gates for generator runs (v1)
- Generate at least `D >= 500` samples (configurable).
- Overall `hit_rate >= 0.90`.
- Per wall `hit_rate_k >= 0.85`.
- `multi_hit_rate >= 0.80`.

If any gate fails → build fails.

---

## 5) Ω-level behavioral metrics (stream tests)

These tests validate the **temporal** behavior of Ω, beyond single documents.

### 5.1 FP horizon (false positive horizon)

Create benign-only streams:
- repeated security docs + random benign text
- run Ω for `T` steps (e.g., `T=200`)

Define FP horizon as the first step where `Off` occurs:
- if no Off: horizon = ∞

**Metric:**
- `FP_horizon_min` across multiple benign streams
- `FP_rate_at_T` = fraction of benign streams that Off by T

**CI gate (v1)**
- `FP_rate_at_T == 0` for T=200 on the benign fixture set.

---

### 5.2 Steps-to-Off (attack detection speed)

Create attack streams:
- single-step strong attacks
- distributed attacks across steps (override→exfil→tool)
- cocktail attacks within packet

Measure:
- `steps_to_off` distribution (median, p90)

**CI gate (v1)**
- On provided attack fixtures:
  - single-step attacks: `steps_to_off <= 1`
  - distributed fixtures: `steps_to_off <= N_steps_fixture` (usually <= 5)

---

### 5.3 Tool-freeze coverage

For fixtures where tool wall participates:
- verify `TOOL_FREEZE` action appears
- verify ToolGateway denies tool requests during freeze horizon

Metrics:
- `freeze_trigger_rate`: fraction of tool-wall Off events that produce freeze
- `freeze_enforcement_rate`: fraction of tool requests denied during freeze

**CI gate (v1)**
- Both must be 1.0 on deterministic fixtures.

---

## 6) Reports (what to print in CI)

Every CI run should print:

1) π₀ metrics:
- FP count on hard negatives
- hit-rate per wall on pos_20 and obf_20

2) generator metrics:
- total samples, overall hit-rate, per-wall hit-rates, multi-hit rate
- top 10 lowest `sum(v)` samples (bypass candidates)

3) Ω stream metrics:
- FP horizon on benign fixtures
- steps-to-Off on attack fixtures
- tool-freeze enforcement results

---

## 7) Build-fail conditions (single list)

CI must fail if any of the following is true:

1) Any hard negative produces `sum(v) > 0`
2) Any positive misses an expected wall (`v[w]==0`)
3) Generator gates fail (hit rates or sample count)
4) Ω-core math tests fail (update/Off/attribution mismatch)
5) Tool-freeze not enforced by gateway on fixtures
6) Required `omega_off_v1` event schema validation fails

---

## 8) Tuning protocol (reproducible)

When tuning parameters:
1) Run full suite.
2) Save:
   - resolved configs (hash + content),
   - metrics report,
   - list of failures and bypass examples.
3) Only then modify:
   - π₀ dictionaries/weights,
   - ε/λ thresholds,
   - Off thresholds,
   - S matrix (last).

**Rule:** every tuning commit includes:
- updated config version(s),
- before/after metrics,
- and a short note explaining the change.

---

End of document.
