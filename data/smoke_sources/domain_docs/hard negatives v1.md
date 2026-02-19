# Hard Negatives v1

This document defines the hard-negative set used to verify that `pi0` does not
trigger on benign security content.

## Goal

For each hard-negative sample:

- expected: `sum(v) == 0` after polarity gating,
- expected: no wall receives positive polarity (`+1`),
- expected: the sample does not contribute to `Off` accumulation.

This protects against false positives on texts that discuss security topics
without malicious intent.

## Canonical Dataset

Primary fixture:

- `tests/data/hard_negatives_50.jsonl`

Schema per row:

```json
{
  "id": "HN-001",
  "family": "hard_negative",
  "targets": [],
  "text": "Do not share API keys or tokens. Keep credentials secure.",
  "expected_nonzero": [],
  "expected_all_zero": true,
  "note": "hard negative"
}
```

## Minimum Test Assertions

1. Project each `text` with `pi0`.
2. Assert `sum(v) == 0` (or `all(v == 0)`).
3. Assert `max(polarity) <= 0`.

## If a Hard Negative Triggers

Most regressions come from one of these:

- override fuzzy matching is too permissive,
- polarity logic misses nearby negation/protective intent around secret nouns,
- tool/evasion markers are not sufficiently intent-gated.

## Related References

- `docs/tests_and_eval.md`
- `tests/test_pi0.py`
- `scripts/run_eval.py`
