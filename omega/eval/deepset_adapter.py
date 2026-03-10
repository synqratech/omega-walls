"""Deepset prompt-injection dataset adapter for Omega evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import pyarrow.parquet as pq
except Exception as exc:  # pragma: no cover - import error path is runtime-dependent
    pq = None
    _PARQUET_IMPORT_ERROR = exc
else:
    _PARQUET_IMPORT_ERROR = None


@dataclass(frozen=True)
class DeepsetSample:
    sample_id: str
    split: str
    text: str
    label: int
    is_attack: bool


@dataclass(frozen=True)
class DeepsetBundle:
    split: str
    mode: str
    total_rows: int
    selected_rows: int
    dropped_empty_text_rows: int
    attack_selected: int
    benign_selected: int
    samples: List[DeepsetSample]


def _require_parquet() -> None:
    if pq is None:
        raise RuntimeError(
            "pyarrow is required for deepset parquet loading. "
            f"Import error: {_PARQUET_IMPORT_ERROR}"
        )


def _resolve_split_parquet(benchmark_root: str, split: str) -> Path:
    root = Path(benchmark_root)
    candidates = sorted((root / "data").glob(f"{split}-*.parquet"))
    if not candidates:
        candidates = sorted(root.glob(f"{split}-*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"Missing deepset parquet for split='{split}' in {root.as_posix()}")
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous deepset parquet for split='{split}': {[c.name for c in candidates]}")
    return candidates[0]


def iter_required_deepset_files(benchmark_root: str) -> Iterable[Path]:
    root = Path(benchmark_root)
    for split in ("train", "test"):
        try:
            yield _resolve_split_parquet(benchmark_root, split)
        except Exception:
            yield root / "data" / f"{split}-*.parquet"
    readme = root / "README.md"
    if readme.exists():
        yield readme


def _load_rows(parquet_path: Path) -> List[Dict[str, object]]:
    _require_parquet()
    table = pq.read_table(parquet_path)
    cols = set(table.column_names)
    required = {"text", "label"}
    if not required.issubset(cols):
        raise ValueError(
            f"Deepset parquet must contain columns {sorted(required)}. "
            f"Got: {sorted(cols)} in {parquet_path.as_posix()}"
        )

    text_values = table["text"].to_pylist()
    label_values = table["label"].to_pylist()
    rows: List[Dict[str, object]] = []
    for idx, (text_val, label_val) in enumerate(zip(text_values, label_values)):
        rows.append(
            {
                "row_idx": idx,
                "text": "" if text_val is None else str(text_val),
                "label": int(label_val),
            }
        )
    return rows


def _pick_rows(rows: List[Dict[str, object]], *, mode: str, max_samples: int, seed: int) -> List[Dict[str, object]]:
    if mode == "full":
        return list(rows)
    if mode != "sampled":
        raise ValueError("deepset mode must be full|sampled")
    if max_samples <= 0 or len(rows) <= max_samples:
        return list(rows)
    rng = random.Random(seed)
    picked = sorted(rng.sample(range(len(rows)), k=max_samples))
    return [rows[i] for i in picked]


def build_deepset_samples(
    benchmark_root: str,
    *,
    split: str = "test",
    mode: str = "full",
    max_samples: int = 116,
    seed: int = 41,
    label_attack_value: int = 1,
) -> DeepsetBundle:
    if split not in {"train", "test"}:
        raise ValueError("deepset split must be train|test")
    parquet_path = _resolve_split_parquet(benchmark_root=benchmark_root, split=split)
    all_rows = _load_rows(parquet_path)
    selected_rows = _pick_rows(all_rows, mode=mode, max_samples=max_samples, seed=seed)

    dropped_empty = 0
    samples: List[DeepsetSample] = []
    attack_selected = 0
    benign_selected = 0
    for row in selected_rows:
        text = str(row.get("text", "")).strip()
        if not text:
            dropped_empty += 1
            continue
        label = int(row["label"])
        is_attack = bool(label == int(label_attack_value))
        if is_attack:
            attack_selected += 1
        else:
            benign_selected += 1
        samples.append(
            DeepsetSample(
                sample_id=f"deepset:{split}:{int(row['row_idx'])}",
                split=split,
                text=text,
                label=label,
                is_attack=is_attack,
            )
        )

    return DeepsetBundle(
        split=split,
        mode=mode,
        total_rows=len(all_rows),
        selected_rows=len(selected_rows),
        dropped_empty_text_rows=dropped_empty,
        attack_selected=attack_selected,
        benign_selected=benign_selected,
        samples=samples,
    )


def deepset_split_stats(benchmark_root: str, split: str) -> Tuple[int, Dict[int, int]]:
    rows = _load_rows(_resolve_split_parquet(benchmark_root=benchmark_root, split=split))
    labels: Dict[int, int] = {}
    for row in rows:
        label = int(row["label"])
        labels[label] = labels.get(label, 0) + 1
    return len(rows), labels
