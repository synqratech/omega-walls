# Omega Walls: Tests & Evaluation (v1)

This document defines:
- which datasets are treated as **ground truth**,
- what metrics we track,
- CI gates (what fails the build),
- and the evaluation workflow to keep tuning reproducible.

For adapter/plugin reliability validation across frameworks, use:
- [Framework Matrix Stand](framework_matrix_stand.md)
- `python scripts/run_framework_matrix_stand.py --layer all --profile dev --strict`

---

## 1) Ground-truth datasets (JSONL)

### 1.1 Canonical test sets (must exist in repo)

```
tests/
  redteam_pos_20.jsonl       # 20 positives (should trigger)
  hard_negatives_50.jsonl    # 50 hard negatives (must NOT trigger)
  redteam_obf_20.jsonl       # 20 obfuscation/paraphrase positives (should trigger)
redteam/
  generator.py               # red-team generator (obf + paraphrase + cocktails + distributed)
```

**Truth rule:** these three JSONL files are the reference truth for v1.

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

## 4) Red-team generator evaluation

### 4.1 Generator output
`redteam/generator.py` produces JSONL-like samples with:
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

---

## 9) Canonical benchmark runner (FW-009)

Use one canonical orchestration entrypoint for reproducible/public comparisons:

```bash
python scripts/run_benchmark.py --dataset-profile core_oss_v1 --mode pi0 --allow-skip-baseline-d
```

Default profile: `core_oss_v1`.

### Artifacts (standardized)

- `artifacts/benchmark/<run_id>/report.json`
- `artifacts/benchmark/<run_id>/scorecard.csv`
- `artifacts/benchmark/<run_id>/dataset_manifest.json`

### Headline comparison policy

- Public scorecard headline is `stateful_target` vs `baseline_d_bare_llm_detector`.
- A/B/C baselines remain in raw report payload for technical controls.
- If baseline D is skipped explicitly (`--allow-skip-baseline-d` with no key), run status is `partial_ok`.

### Reproducibility contract

- Dataset provenance is strict and mandatory per run (`dataset_manifest.json`).
- Each dataset row includes path, source type, source URL, sha256, and deterministic stats.
- Missing or incomplete manifest data invalidates the run (`failed_reproducibility`).

---

## 10) Docker quickstart for API eval surface (FW-008)

For pilot/devrel onboarding, use the official API-only container path.

Run container:

```bash
docker run --rm -p 8080:8080 ghcr.io/<owner>/<repo>/omega-walls-api:latest
```

Verify health:

```bash
curl -fsS http://127.0.0.1:8080/healthz
```

Run scan smoke:

```bash
curl -fsS \
  -H "Content-Type: application/json" \
  -H "X-API-Key: quickstart-api-key" \
  -d '{"tenant_id":"eval-smoke","request_id":"req-1","extracted_text":"Normal support text."}' \
  http://127.0.0.1:8080/v1/scan/attachment
```

Notes:
- quickstart profile is the frictionless demo path (`X-API-Key: quickstart-api-key`, no HMAC).
- strict eval/pilot path should run `--profile dev|pilot` with `OMEGA_API_HMAC_SECRET` and signed headers.

---

## 10) FW-001 release gate (coverage + perf)

FW-001 is a mandatory PR-blocking gate:

- workflow: `.github/workflows/fw001-release-gate.yml`
- runner: `ubuntu-latest`, `Python 3.13`
- coverage gate: `pytest ... --cov=omega --cov-branch --cov-fail-under=85`
- perf gate: `python scripts/check_fw001_perf_gate.py --strict --perf-overhead-max 0.15`

Frozen perf baseline file:

- `config/perf_baselines/fw001_pi0_py313_ubuntu.json`

### How perf pass/fail is computed

For each size bucket `short|medium|large`:

```text
overhead_ratio = (candidate_p95_ms - baseline_p95_ms) / baseline_p95_ms
```

Gate passes only if every bucket satisfies:

```text
overhead_ratio <= 0.15
```

### Baseline update policy

- Baseline is not auto-updated in regular PRs.
- Baseline changes must be made in a dedicated, explicit PR with rationale and before/after artifacts.

---

## 11) Real-workflow validation stand (LangChain + OpenClaw)

Run:

```bash
python scripts/run_real_agent_stand.py --profile dev --strict
```

Artifacts:

- `artifacts/real_agent_stand/<run_id>/report.json`
- `artifacts/real_agent_stand/<run_id>/phase1_langchain.json`
- `artifacts/real_agent_stand/<run_id>/api.log`

Gate focus:

- blocked input/tool visibility in real workflow hooks
- require-approval signal observed
- webfetch guard path observed
- fail-closed outage behavior
- `orphan_executions == 0` and `gateway_coverage >= 1.0`
