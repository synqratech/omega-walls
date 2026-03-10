from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from omega.eval.deepset_adapter import build_deepset_samples


def _write_split(path: Path, rows: list[dict]) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table(
        {
            "text": [row["text"] for row in rows],
            "label": [row["label"] for row in rows],
        }
    )
    pq.write_table(table, path.as_posix())


def _mk_local_tmp(name: str) -> Path:
    root = Path("tests/_tmp")
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"{name}-{uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _mk_dataset() -> str:
    root = _mk_local_tmp("deepset-adapter") / "deepset"
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_split(
        data_dir / "train-00000.parquet",
        [
            {"text": "benign train", "label": 0},
            {"text": "attack train", "label": 1},
        ],
    )
    _write_split(
        data_dir / "test-00000.parquet",
        [
            {"text": "benign test 1", "label": 0},
            {"text": "attack test 1", "label": 1},
            {"text": "attack test 2", "label": 1},
            {"text": "  ", "label": 0},
        ],
    )
    return root.as_posix()


def test_deepset_adapter_sampled_is_deterministic():
    root = _mk_dataset()
    a = build_deepset_samples(
        benchmark_root=root,
        split="test",
        mode="sampled",
        max_samples=2,
        seed=41,
        label_attack_value=1,
    )
    b = build_deepset_samples(
        benchmark_root=root,
        split="test",
        mode="sampled",
        max_samples=2,
        seed=41,
        label_attack_value=1,
    )
    assert [s.sample_id for s in a.samples] == [s.sample_id for s in b.samples]


def test_deepset_adapter_label_mapping_and_drop_empty():
    root = _mk_dataset()
    bundle = build_deepset_samples(
        benchmark_root=root,
        split="test",
        mode="full",
        max_samples=100,
        seed=7,
        label_attack_value=1,
    )
    assert bundle.total_rows == 4
    assert bundle.selected_rows == 4
    assert bundle.dropped_empty_text_rows == 1
    assert bundle.attack_selected == 2
    assert bundle.benign_selected == 1
