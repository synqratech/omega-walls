"""Gold-slice utilities for ordinal/polarity annotation agreement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from omega.interfaces.contracts_v1 import WALLS_V1

WALLS = list(WALLS_V1)


def _jsonl_rows(path: Path) -> Iterable[Dict[str, Any]]:
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            yield row


def _validate_record(row: Mapping[str, Any]) -> None:
    required = {
        "sample_id",
        "text",
        "wall_labels",
        "pressure_level",
        "polarity",
        "source",
        "source_type",
        "source_trust",
        "chunk_bucket",
    }
    missing = sorted(required.difference(row.keys()))
    if missing:
        raise ValueError(f"gold-slice row missing required keys: {missing}")
    sample_id = str(row.get("sample_id", "")).strip()
    if not sample_id:
        raise ValueError("gold-slice sample_id must be non-empty")
    if not str(row.get("text", "")).strip():
        raise ValueError(f"gold-slice text must be non-empty for sample_id={sample_id}")
    wall = [int(x) for x in list(row.get("wall_labels", []))]
    pressure = [int(x) for x in list(row.get("pressure_level", []))]
    polarity = [int(x) for x in list(row.get("polarity", []))]
    if len(wall) != 4 or len(pressure) != 4 or len(polarity) != 4:
        raise ValueError("wall_labels/pressure_level/polarity must have length 4")
    for i in range(4):
        if int(wall[i]) not in {0, 1}:
            raise ValueError("wall_labels values must be 0|1")
        if int(pressure[i]) not in {0, 1, 2, 3}:
            raise ValueError("pressure_level values must be 0..3")
        if int(polarity[i]) not in {-1, 0, 1}:
            raise ValueError("polarity values must be -1|0|1")
        if int(wall[i]) == 0 and (int(pressure[i]) != 0 or int(polarity[i]) != 0):
            raise ValueError("gold-slice rule violation: wall=0 requires pressure=0 and polarity=0")


def load_gold_slice_jsonl(path: str) -> List[Dict[str, Any]]:
    src = Path(path)
    if not src.exists():
        raise ValueError(f"gold-slice file not found: {src.as_posix()}")
    out: List[Dict[str, Any]] = []
    for row in _jsonl_rows(src):
        _validate_record(row)
        out.append(dict(row))
    if not out:
        raise ValueError("gold-slice file is empty")
    return out


def quadratic_weighted_kappa(y_true: Sequence[int], y_pred: Sequence[int], classes: int) -> float:
    n = len(y_true)
    if n == 0 or n != len(y_pred):
        return 0.0
    c = max(2, int(classes))
    obs = [[0.0 for _ in range(c)] for _ in range(c)]
    hist_true = [0.0] * c
    hist_pred = [0.0] * c
    for a_raw, b_raw in zip(y_true, y_pred):
        a = min(max(int(a_raw), 0), c - 1)
        b = min(max(int(b_raw), 0), c - 1)
        obs[a][b] += 1.0
        hist_true[a] += 1.0
        hist_pred[b] += 1.0
    for i in range(c):
        for j in range(c):
            obs[i][j] /= float(n)
    exp = [[(hist_true[i] * hist_pred[j]) / float(n * n) for j in range(c)] for i in range(c)]

    denom_base = float((c - 1) * (c - 1))
    num = 0.0
    den = 0.0
    for i in range(c):
        for j in range(c):
            w = float((i - j) * (i - j)) / denom_base
            num += w * obs[i][j]
            den += w * exp[i][j]
    if den <= 1e-12:
        return 1.0 if num <= 1e-12 else 0.0
    return float(1.0 - (num / den))


def compute_gold_slice_agreement(
    rows_a: Sequence[Mapping[str, Any]],
    rows_b: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    map_a = {str(r["sample_id"]): dict(r) for r in rows_a}
    map_b = {str(r["sample_id"]): dict(r) for r in rows_b}
    ids_a = set(map_a.keys())
    ids_b = set(map_b.keys())
    matched_ids = sorted(ids_a.intersection(ids_b))
    only_a = sorted(ids_a.difference(ids_b))
    only_b = sorted(ids_b.difference(ids_a))

    ord_a = [[] for _ in range(4)]
    ord_b = [[] for _ in range(4)]
    pol_a = [[] for _ in range(4)]
    pol_b = [[] for _ in range(4)]
    exact_match = 0
    adjudication_rows: List[Dict[str, Any]] = []

    for sample_id in matched_ids:
        ra = map_a[sample_id]
        rb = map_b[sample_id]
        pa = [int(x) for x in ra["pressure_level"]]
        pb = [int(x) for x in rb["pressure_level"]]
        la = [int(x) for x in ra["polarity"]]
        lb = [int(x) for x in rb["polarity"]]
        wa = [int(x) for x in ra["wall_labels"]]
        wb = [int(x) for x in rb["wall_labels"]]
        for i in range(4):
            ord_a[i].append(pa[i])
            ord_b[i].append(pb[i])
            pol_a[i].append(la[i] + 1)  # map -1|0|1 -> 0|1|2 for kappa
            pol_b[i].append(lb[i] + 1)
        is_exact = pa == pb and la == lb and wa == wb
        if is_exact:
            exact_match += 1
            continue
        wall_breakdown: Dict[str, Any] = {}
        severity = 0
        for i, wall in enumerate(WALLS):
            ord_delta = abs(int(pa[i]) - int(pb[i]))
            pol_delta = abs(int(la[i]) - int(lb[i]))
            wall_delta = ord_delta + pol_delta + abs(int(wa[i]) - int(wb[i]))
            severity += wall_delta
            wall_breakdown[wall] = {
                "wall_label_a": int(wa[i]),
                "wall_label_b": int(wb[i]),
                "pressure_a": int(pa[i]),
                "pressure_b": int(pb[i]),
                "polarity_a": int(la[i]),
                "polarity_b": int(lb[i]),
                "delta": int(wall_delta),
            }
        adjudication_rows.append(
            {
                "sample_id": sample_id,
                "text": str(ra.get("text", "")),
                "source": str(ra.get("source", "")),
                "source_type": str(ra.get("source_type", "")),
                "source_trust": str(ra.get("source_trust", "")),
                "chunk_bucket": str(ra.get("chunk_bucket", "")),
                "severity": int(severity),
                "wall_breakdown": wall_breakdown,
            }
        )

    agreement = {
        "matched_count": int(len(matched_ids)),
        "exact_match_rate": float(exact_match) / float(len(matched_ids)) if matched_ids else 0.0,
        "ordinal_quadratic_kappa_per_wall": {
            WALLS[i]: quadratic_weighted_kappa(ord_a[i], ord_b[i], classes=4) for i in range(4)
        },
        "polarity_quadratic_kappa_per_wall": {
            WALLS[i]: quadratic_weighted_kappa(pol_a[i], pol_b[i], classes=3) for i in range(4)
        },
        "only_in_annotator_a": int(len(only_a)),
        "only_in_annotator_b": int(len(only_b)),
    }
    adjudication_rows.sort(key=lambda x: int(x.get("severity", 0)), reverse=True)
    return agreement, adjudication_rows

