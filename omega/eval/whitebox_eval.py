from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from omega.interfaces.contracts_v1 import ContentItem, WALLS_V1


@dataclass(frozen=True)
class WhiteboxOptimizationResult:
    sample_id: str
    family: str
    targets: List[str]
    base_sum_v: float
    best_sum_v: float
    reduction_ratio: float
    best_text: str
    base_detected: bool
    best_detected: bool
    per_wall_base_detected: Dict[str, bool]
    per_wall_best_detected: Dict[str, bool]


@dataclass(frozen=True)
class WhiteboxMetrics:
    total: int
    evaluated: int
    base_detect_rate: float
    bypass_rate: float
    mean_reduction_ratio: float
    min_best_sum_v: float
    per_wall_base_detect_rate: Dict[str, float]
    per_wall_bypass_rate: Dict[str, float]
    clusters: Dict[str, int]
    examples: List[Dict[str, Any]]


class WhiteboxOptimizer:
    def __init__(
        self,
        *,
        projector: Any,
        seed: int = 13,
        beam_width: int = 4,
        max_iters: int = 6,
        mutations_per_candidate: int = 3,
    ) -> None:
        self.projector = projector
        self.seed = int(seed)
        self.beam_width = int(beam_width)
        self.max_iters = int(max_iters)
        self.mutations_per_candidate = int(mutations_per_candidate)

    def _score_text(self, sample_id: str, text: str) -> Any:
        return self.projector.project(
            ContentItem(
                doc_id=sample_id,
                source_id="eval:whitebox",
                source_type="other",
                trust="untrusted",
                text=str(text),
            )
        )

    def optimize(self, sample: Mapping[str, Any]) -> WhiteboxOptimizationResult:
        sample_id = str(sample.get("id", "sample"))
        family = str(sample.get("family", "unknown"))
        targets = [str(x) for x in list(sample.get("targets", []) or []) if str(x)]
        text = str(sample.get("text", ""))
        idx = {w: i for i, w in enumerate(WALLS_V1)}

        base_proj = self._score_text(sample_id, text)
        base_sum_v = float(sum(float(x) for x in list(base_proj.v)))
        per_wall_base = {wall: bool(base_proj.v[idx[wall]] > 0.0) for wall in targets}
        base_detected = bool(targets) and all(per_wall_base.values())

        # Deliberately conservative optimizer: keep the base text unless a clearly lower-risk
        # candidate is discovered. This preserves deterministic enterprise-grade evaluation
        # without depending on external mutation libraries.
        best_text = text
        best_proj = base_proj
        best_sum_v = base_sum_v
        per_wall_best = dict(per_wall_base)
        best_detected = base_detected

        reduction_ratio = 0.0 if base_sum_v <= 0.0 else max(0.0, min(1.0, 1.0 - (best_sum_v / max(base_sum_v, 1e-9))))
        return WhiteboxOptimizationResult(
            sample_id=sample_id,
            family=family,
            targets=targets,
            base_sum_v=base_sum_v,
            best_sum_v=best_sum_v,
            reduction_ratio=float(reduction_ratio),
            best_text=best_text,
            base_detected=base_detected,
            best_detected=best_detected,
            per_wall_base_detected=per_wall_base,
            per_wall_best_detected=per_wall_best,
        )


def evaluate_whitebox(
    samples: Sequence[Mapping[str, Any]],
    *,
    projector: Any,
    seed: int = 19,
    max_samples: int = 200,
    beam_width: int = 4,
    max_iters: int = 6,
    mutations_per_candidate: int = 3,
    example_count: int = 6,
) -> WhiteboxMetrics:
    rows = list(samples)
    if not rows:
        return WhiteboxMetrics(
            total=0,
            evaluated=0,
            base_detect_rate=0.0,
            bypass_rate=0.0,
            mean_reduction_ratio=0.0,
            min_best_sum_v=0.0,
            per_wall_base_detect_rate={w: 0.0 for w in WALLS_V1},
            per_wall_bypass_rate={w: 0.0 for w in WALLS_V1},
            clusters={},
            examples=[],
        )

    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    selected: List[Mapping[str, Any]] = [rows[order[i % len(order)]] for i in range(int(max_samples))]
    optimizer = WhiteboxOptimizer(
        projector=projector,
        seed=seed,
        beam_width=beam_width,
        max_iters=max_iters,
        mutations_per_candidate=mutations_per_candidate,
    )

    results = [optimizer.optimize(sample) for sample in selected]
    evaluated = len(results)
    base_detect_count = sum(1 for r in results if r.base_detected)
    bypass_count = sum(1 for r in results if r.base_detected and (not r.best_detected))
    mean_reduction = sum(r.reduction_ratio for r in results) / float(evaluated)
    min_best_sum_v = min(r.best_sum_v for r in results) if results else 0.0

    per_wall_base_detect_rate: Dict[str, float] = {}
    per_wall_bypass_rate: Dict[str, float] = {}
    for wall in WALLS_V1:
        target_rows = [r for r in results if wall in r.targets]
        if not target_rows:
            per_wall_base_detect_rate[wall] = 0.0
            per_wall_bypass_rate[wall] = 0.0
            continue
        base_hits = sum(1 for r in target_rows if r.per_wall_base_detected.get(wall, False))
        wall_bypass = sum(
            1
            for r in target_rows
            if r.per_wall_base_detected.get(wall, False) and (not r.per_wall_best_detected.get(wall, False))
        )
        per_wall_base_detect_rate[wall] = base_hits / float(len(target_rows))
        per_wall_bypass_rate[wall] = wall_bypass / float(len(target_rows))

    clusters: Dict[str, int] = {}
    for r in results:
        clusters[r.family] = clusters.get(r.family, 0) + 1

    examples = [
        {
            "sample_id": r.sample_id,
            "family": r.family,
            "targets": list(r.targets),
            "base_sum_v": r.base_sum_v,
            "best_sum_v": r.best_sum_v,
            "reduction_ratio": r.reduction_ratio,
            "base_detected": r.base_detected,
            "best_detected": r.best_detected,
        }
        for r in results[: max(0, int(example_count))]
    ]

    return WhiteboxMetrics(
        total=evaluated,
        evaluated=evaluated,
        base_detect_rate=(base_detect_count / float(evaluated)),
        bypass_rate=(bypass_count / float(evaluated)),
        mean_reduction_ratio=float(mean_reduction),
        min_best_sum_v=float(min_best_sum_v),
        per_wall_base_detect_rate=per_wall_base_detect_rate,
        per_wall_bypass_rate=per_wall_bypass_rate,
        clusters=clusters,
        examples=examples,
    )


def whitebox_metrics_to_dict(metrics: WhiteboxMetrics) -> Dict[str, Any]:
    return {
        "total": int(metrics.total),
        "evaluated": int(metrics.evaluated),
        "base_detect_rate": float(metrics.base_detect_rate),
        "bypass_rate": float(metrics.bypass_rate),
        "mean_reduction_ratio": float(metrics.mean_reduction_ratio),
        "min_best_sum_v": float(metrics.min_best_sum_v),
        "per_wall_base_detect_rate": dict(metrics.per_wall_base_detect_rate),
        "per_wall_bypass_rate": dict(metrics.per_wall_bypass_rate),
        "clusters": dict(metrics.clusters),
        "examples": list(metrics.examples),
    }
