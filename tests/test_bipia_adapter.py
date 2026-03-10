from __future__ import annotations

from omega.eval.bipia_adapter import build_bipia_task_bundles


FIXTURE_ROOT = "tests/data/bipia_fixture/benchmark"


def test_bipia_adapter_sampled_is_deterministic():
    a = build_bipia_task_bundles(
        benchmark_root=FIXTURE_ROOT,
        split="test",
        mode="sampled",
        max_contexts_per_task=1,
        max_attacks_per_task=10,
        seed=41,
    )
    b = build_bipia_task_bundles(
        benchmark_root=FIXTURE_ROOT,
        split="test",
        mode="sampled",
        max_contexts_per_task=1,
        max_attacks_per_task=10,
        seed=41,
    )
    for task in a:
        assert [s.sample_id for s in a[task].attack_samples] == [s.sample_id for s in b[task].attack_samples]
        assert [s.sample_id for s in a[task].benign_samples] == [s.sample_id for s in b[task].benign_samples]


def test_bipia_adapter_counts_from_fixture():
    bundles = build_bipia_task_bundles(
        benchmark_root=FIXTURE_ROOT,
        split="test",
        mode="sampled",
        max_contexts_per_task=1,
        max_attacks_per_task=10,
        seed=7,
    )
    # Text tasks: 2 attacks * 3 positions = 6 attack samples; 1 benign sample.
    for task in ("email", "table", "qa", "abstract"):
        assert len(bundles[task].attack_samples) == 6
        assert len(bundles[task].benign_samples) == 1
    # Code task: 1 attack * 3 positions = 3 attack samples; 1 benign sample.
    assert len(bundles["code"].attack_samples) == 3
    assert len(bundles["code"].benign_samples) == 1
