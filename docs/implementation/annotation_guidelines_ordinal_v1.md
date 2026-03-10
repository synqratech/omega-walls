# Annotation Guidelines: Ordinal + Polarity (v1)

## Purpose
This guide defines how to manually label the `gold_slice.jsonl` set for PiTheta calibration.

## Target Schema
Each record must include:
1. `sample_id`
2. `text`
3. `wall_labels` (4 ints, 0|1)
4. `pressure_level` (4 ints, 0..3)
5. `polarity` (4 ints, -1|0|1)
6. `source`, `source_type`, `source_trust`
7. `chunk_bucket` (`64|128_256|512`) and `approx_tokens`

Wall order is fixed:
1. `override_instructions`
2. `secret_exfiltration`
3. `tool_or_action_abuse`
4. `policy_evasion`

## Hard Label Rule
If `wall_labels[k] == 0`, then:
1. `pressure_level[k] = 0`
2. `polarity[k] = 0` (neutral)

This rule is mandatory to prevent polarity class drift.

## Pressure Scale (Ordinal)
For each wall:
1. `0`: no actionable threat signal
2. `1`: weak/ambiguous threat signal
3. `2`: clear threat instruction
4. `3`: strong/direct threat imperative

## Polarity Scale
For each wall:
1. `-1`: protective (explicitly blocks harmful action)
2. `0`: neutral (mentions topic, no harmful directive)
3. `+1`: directive (pushes toward harmful action)

## Context Rules
Use source context while labeling:
1. `source_type=doc/wiki` with defensive language is often neutral/protective.
2. `source_type=user_input/web/email/ticket` with imperatives is often directive.
3. Do not label by keywords only; label by intent in context.

## Chunking Policy for Gold Slice
Keep a fixed mix:
1. ~30% short chunks (`chunk_bucket=64`)
2. ~50% medium chunks (`chunk_bucket=128_256`)
3. ~20% long chunks (`chunk_bucket=512`)

Use `approx_tokens` as approximate token count from text length.

## Annotation Workflow
1. Two annotators label the same slice independently.
2. Run:
```bash
python scripts/gold_slice_kappa.py --annotator-a <A.jsonl> --annotator-b <B.jsonl>
```
3. Review `adjudication_report.jsonl` sorted by severity.
4. Resolve disagreements and produce final `gold_slice.jsonl`.

## Agreement Targets
1. Ordinal quadratic kappa per wall: `>= 0.70`
2. Polarity quadratic kappa per wall: `>= 0.65`
3. Exact match rate: monitor trend; do not use as the only quality signal.

## Minimum Gold Slice Size
Before cloud fine-tune:
1. Minimum: `150` examples
2. Preferred: `200-300` examples

