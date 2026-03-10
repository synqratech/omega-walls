from __future__ import annotations

from pathlib import Path

from scripts.build_strict_pi_holdout import build_strict_holdout


def test_strict_holdout_builder_is_deterministic():
    root = Path(__file__).resolve().parent.parent
    a = build_strict_holdout(repo_root=root, seed=41)
    b = build_strict_holdout(repo_root=root, seed=41)
    assert [(x.id, x.label, x.family, x.text) for x in a] == [(x.id, x.label, x.family, x.text) for x in b]


def test_strict_holdout_builder_family_coverage():
    root = Path(__file__).resolve().parent.parent
    rows = build_strict_holdout(repo_root=root, seed=41)
    fam_attack = {r.family for r in rows if int(r.label) == 1}
    fam_benign = {r.family for r in rows if int(r.label) == 0}
    assert {"override", "roleplay", "leak", "sql", "tool", "obfuscation"}.issubset(fam_attack)
    assert {"benign_defensive", "benign_task"}.issubset(fam_benign)
    assert all(not bool(r.disputed) for r in rows)
