from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from omega.config.loader import load_resolved_config
from omega.interfaces.contracts_v1 import ContentItem, WALLS_V1
from omega.projector.pi0_intent_v2 import Pi0IntentAwareV2
from redteam.generator import generate


@dataclass
class EvalMetrics:
    total: int
    overall_hit_rate: float
    per_wall_hit_rate: Dict[str, float]
    multi_hit_rate: float



def _wall_idx() -> Dict[str, int]:
    return {w: i for i, w in enumerate(WALLS_V1)}


def evaluate_samples(samples: Iterable[Dict], projector: Pi0IntentAwareV2) -> EvalMetrics:
    samples = list(samples)
    idx = _wall_idx()
    targeted = {w: 0 for w in WALLS_V1}
    hits = {w: 0 for w in WALLS_V1}
    total_hits = 0.0

    cocktail_total = 0
    cocktail_hits = 0

    for sample in samples:
        item = ContentItem(
            doc_id=sample.get("id", "sample"),
            source_id="synthetic:redteam",
            source_type="other",
            trust="untrusted",
            text=sample["text"],
        )
        proj = projector.project(item)
        targets = sample.get("targets", [])
        detected_targets = 0
        for wall in targets:
            targeted[wall] += 1
            if proj.v[idx[wall]] > 0:
                hits[wall] += 1
                detected_targets += 1
        if targets:
            total_hits += detected_targets / len(targets)

        if len(targets) >= 2:
            cocktail_total += 1
            if sum(1 for value in proj.v if value > 0) >= 2:
                cocktail_hits += 1

    per_wall = {w: (hits[w] / targeted[w] if targeted[w] else 1.0) for w in WALLS_V1}
    return EvalMetrics(
        total=len(samples),
        overall_hit_rate=(total_hits / len(samples) if samples else 0.0),
        per_wall_hit_rate=per_wall,
        multi_hit_rate=(cocktail_hits / cocktail_total if cocktail_total else 1.0),
    )


def evaluate_generated(seed: int = 7, n_per_family: int = 200) -> EvalMetrics:
    cfg = load_resolved_config(profile="dev").resolved
    projector = Pi0IntentAwareV2(cfg)
    samples = [asdict(s) for s in generate(seed=seed, n_per_family=n_per_family)]
    return evaluate_samples(samples, projector)


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows
