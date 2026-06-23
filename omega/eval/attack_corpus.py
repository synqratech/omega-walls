from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SESSION_PACK = (
    REPO_ROOT / "tests" / "data" / "session_benchmark" / "redteam_cross_source_polish_ab_n24_session_pack.jsonl"
)
FAMILY_TARGETS: Dict[str, List[str]] = {
    "xsrc_policy_tool": ["policy_evasion", "tool_or_action_abuse"],
    "xsrc_secret_policy": ["secret_exfiltration", "policy_evasion"],
    "xsrc_override_tool": ["override_instructions", "tool_or_action_abuse"],
    "xsrc_override_secret": ["override_instructions", "secret_exfiltration"],
}


@dataclass(frozen=True)
class SourceFragment:
    phase: str
    source_type: str
    trust: str
    session_id: str
    turn_id: int
    text: str
    temporal_gap_from_prev_min: int = 0


@dataclass(frozen=True)
class AttackSample:
    id: str
    family: str
    text: str
    targets: List[str]
    attack_type: str = "cross_source_session_pack"
    source_fragments: List[SourceFragment] = field(default_factory=list)
    trigger: str = ""
    detection_rationale: str = ""
    quality_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratedMetrics:
    total: int
    overall_hit_rate: float
    per_wall_hit_rate: Dict[str, float]
    multi_hit_rate: float


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _source_trust(source_type: str) -> str:
    source = str(source_type or "").strip().lower()
    if source in {"trusted", "memory_trusted", "internal_trusted"}:
        return "trusted"
    if source in {"semi_trusted", "internal"}:
        return "semi_trusted"
    return "untrusted"


def _phase_from_source_ref(source_ref: str) -> str:
    parts = str(source_ref or "").split(":")
    if len(parts) >= 2 and parts[1]:
        return str(parts[1])
    return "unknown"


def _sample_id_from_row(row: Mapping[str, Any]) -> str:
    source_ref = str(row.get("source_ref", "")).strip()
    if source_ref:
        return source_ref.split(":", 1)[0]
    return str(row.get("session_id", "unknown"))


def _group_attack_rows(pack_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in load_jsonl(pack_path):
        if str(row.get("label_session", "")).strip().lower() != "attack":
            continue
        sample_id = _sample_id_from_row(row)
        grouped.setdefault(sample_id, []).append(row)
    return grouped


def _rows_to_sample(sample_id: str, rows: Sequence[Mapping[str, Any]]) -> AttackSample:
    ordered = sorted(
        rows,
        key=lambda r: (
            str(r.get("session_id", "")),
            int(r.get("turn_id", 0) or 0),
            str(r.get("source_ref", "")),
        ),
    )
    family = str(ordered[0].get("family", "unknown")).strip()
    fragments: List[SourceFragment] = []
    prev_session = None
    prev_turn = None
    for row in ordered:
        session_id = str(row.get("session_id", ""))
        turn_id = int(row.get("turn_id", 0) or 0)
        if prev_session is None:
            gap = 0
        elif session_id != prev_session:
            gap = 45
        else:
            gap = max(5, (turn_id - int(prev_turn or 0)) * 5)
        prev_session = session_id
        prev_turn = turn_id
        fragments.append(
            SourceFragment(
                phase=_phase_from_source_ref(str(row.get("source_ref", ""))),
                source_type=str(row.get("source_type", "other")),
                trust=_source_trust(str(row.get("source_type", ""))),
                session_id=session_id,
                turn_id=turn_id,
                text=str(row.get("text", "")),
                temporal_gap_from_prev_min=int(gap),
            )
        )
    text = "\n".join(str(f.text).strip() for f in fragments if str(f.text).strip())
    quality_meta = {
        "gate_passed": True,
        "generation_mode": "session_pack",
        "source": "session_pack",
        "llm_used": False,
        "fragment_count": len(fragments),
        "family_targets": list(FAMILY_TARGETS.get(family, [])),
    }
    return AttackSample(
        id=str(sample_id),
        family=family,
        text=text,
        targets=list(FAMILY_TARGETS.get(family, [])),
        source_fragments=fragments,
        trigger=str(fragments[-1].phase if fragments else "unknown"),
        detection_rationale=f"{family} operational multi-source attack from local session pack",
        quality_meta=quality_meta,
    )


def load_attack_samples_from_session_pack(pack_path: str | Path = DEFAULT_SESSION_PACK) -> List[AttackSample]:
    grouped = _group_attack_rows(Path(pack_path))
    samples = [_rows_to_sample(sample_id, rows) for sample_id, rows in grouped.items()]
    samples.sort(key=lambda s: (s.family, s.id))
    return samples


def generate(seed: int = 7, n_per_family: int = 200, pack_path: str | Path = DEFAULT_SESSION_PACK) -> List[AttackSample]:
    base_samples = load_attack_samples_from_session_pack(pack_path=pack_path)
    by_family: Dict[str, List[AttackSample]] = {}
    for sample in base_samples:
        by_family.setdefault(sample.family, []).append(sample)

    out: List[AttackSample] = []
    for family in sorted(FAMILY_TARGETS):
        family_samples = list(by_family.get(family, []))
        if not family_samples:
            continue
        rng = random.Random(f"{seed}:{family}")
        rng.shuffle(family_samples)
        for idx in range(int(n_per_family)):
            base = family_samples[idx % len(family_samples)]
            replica_idx = idx + 1
            out.append(
                AttackSample(
                    id=f"{base.id}-R{replica_idx:03d}",
                    family=base.family,
                    text=base.text,
                    targets=list(base.targets),
                    attack_type=base.attack_type,
                    source_fragments=list(base.source_fragments),
                    trigger=base.trigger,
                    detection_rationale=base.detection_rationale,
                    quality_meta={
                        **dict(base.quality_meta),
                        "replica_index": replica_idx,
                        "base_sample_id": base.id,
                    },
                )
            )
    return out


def evaluate_generated(*, projector: Any, seed: int = 7, n_per_family: int = 200) -> GeneratedMetrics:
    samples = generate(seed=seed, n_per_family=n_per_family)
    if not samples:
        return GeneratedMetrics(total=0, overall_hit_rate=0.0, per_wall_hit_rate={}, multi_hit_rate=0.0)

    from omega.interfaces.contracts_v1 import ContentItem, WALLS_V1

    idx = {w: i for i, w in enumerate(WALLS_V1)}
    wall_hits = {w: 0 for w in WALLS_V1}
    wall_total = {w: 0 for w in WALLS_V1}
    sample_hits = 0
    multi_hits = 0

    for sample in samples:
        proj = projector.project(
            ContentItem(
                doc_id=sample.id,
                source_id="eval:session_pack",
                source_type="other",
                trust="untrusted",
                text=sample.text,
            )
        )
        detected_targets = 0
        ok = True
        for wall in sample.targets:
            wall_total[wall] += 1
            if proj.v[idx[wall]] > 0:
                wall_hits[wall] += 1
                detected_targets += 1
            else:
                ok = False
        if ok:
            sample_hits += 1
        if detected_targets >= min(2, len(sample.targets)):
            multi_hits += 1

    per_wall = {wall: (wall_hits[wall] / wall_total[wall] if wall_total[wall] else 1.0) for wall in WALLS_V1}
    return GeneratedMetrics(
        total=len(samples),
        overall_hit_rate=(sample_hits / len(samples)),
        per_wall_hit_rate=per_wall,
        multi_hit_rate=(multi_hits / len(samples)),
    )
