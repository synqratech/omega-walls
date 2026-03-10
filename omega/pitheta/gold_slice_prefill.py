"""Gold-slice prefill builder (source/chunk stratified, deterministic)."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from omega.eval.deepset_adapter import build_deepset_samples
from omega.interfaces.contracts_v1 import ContentItem, WALLS_V1
from omega.projector.pi0_intent_v2 import Pi0IntentAwareV2
from redteam.generator import generate

WALLS = list(WALLS_V1)
SOURCE_DEFAULTS = {
    "deepset_train": {"source_type": "user_input", "source_trust": "untrusted"},
    "wainject_text": {"source_type": "web", "source_trust": "untrusted"},
    "redteam_synth_train": {"source_type": "user_input", "source_trust": "untrusted"},
}


@dataclass(frozen=True)
class PrefillConfig:
    target_size: int
    seed: int
    deepset_root: str
    wainject_root: str
    source_weights: Dict[str, float]
    chunk_weights: Dict[str, float]
    ordinal_bins: Tuple[float, float, float]
    profile: str


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _word_tokens(text: str) -> int:
    return max(1, len([tok for tok in str(text).strip().split() if tok]))


def _chunk_bucket(n_tokens: int) -> str:
    n = max(1, int(n_tokens))
    if n <= 96:
        return "64"
    if n <= 320:
        return "128_256"
    return "512"


def _expand_text_to_bucket(text: str, bucket: str) -> Tuple[str, int]:
    text_norm = str(text).strip()
    if not text_norm:
        return text_norm, 0
    target_min = {"64": 1, "128_256": 128, "512": 512}.get(str(bucket), 1)
    target_max = {"64": 96, "128_256": 256, "512": 640}.get(str(bucket), 96)
    pieces = [text_norm]
    tokens = _word_tokens(text_norm)
    while tokens < int(target_min):
        pieces.append(text_norm)
        joined = " ".join(pieces)
        tokens = _word_tokens(joined)
        if len(pieces) > 64:
            break
    joined = " ".join(pieces)
    toks = _word_tokens(joined)
    if toks > int(target_max):
        words = [w for w in joined.split() if w]
        joined = " ".join(words[: int(target_max)])
        toks = _word_tokens(joined)
    return joined, toks


def _rebalance_chunk_buckets(
    rows: List[Dict[str, Any]],
    *,
    target_size: int,
    chunk_weights: Mapping[str, float],
    seed: int,
) -> List[Dict[str, Any]]:
    if not rows:
        return rows
    desired = _allocate_counts(int(target_size), chunk_weights, ["64", "128_256", "512"])
    by_bucket: Dict[str, List[int]] = {"64": [], "128_256": [], "512": []}
    for idx, row in enumerate(rows):
        bucket = str(row.get("chunk_bucket", "64"))
        if bucket not in by_bucket:
            bucket = "64"
        by_bucket[bucket].append(idx)

    rng = random.Random(seed + 4441)
    # Move rows from 64->128_256 and 64/128_256->512 by text expansion.
    for target_bucket in ("128_256", "512"):
        deficit = int(desired.get(target_bucket, 0)) - len(by_bucket[target_bucket])
        if deficit <= 0:
            continue
        donor_order = ["64", "128_256"] if target_bucket == "512" else ["64"]
        for donor in donor_order:
            if deficit <= 0:
                break
            donor_idx = list(by_bucket[donor])
            rng.shuffle(donor_idx)
            for idx in donor_idx:
                if deficit <= 0:
                    break
                row = rows[idx]
                text_new, toks = _expand_text_to_bucket(str(row.get("text", "")), target_bucket)
                if toks <= 0:
                    continue
                row["text"] = text_new
                row["approx_tokens"] = int(toks)
                row["chunk_bucket"] = str(target_bucket)
                by_bucket[donor] = [x for x in by_bucket[donor] if x != idx]
                by_bucket[target_bucket].append(idx)
                deficit -= 1
    return rows


def _normalize_weights(weights: Mapping[str, float], keys: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    total = 0.0
    for k in keys:
        v = max(0.0, float(weights.get(k, 0.0)))
        out[str(k)] = v
        total += v
    if total <= 0:
        v = 1.0 / float(max(1, len(keys)))
        return {str(k): v for k in keys}
    return {str(k): float(out[str(k)]) / total for k in keys}


def _allocate_counts(total: int, weights: Mapping[str, float], keys: Sequence[str]) -> Dict[str, int]:
    if total <= 0:
        return {str(k): 0 for k in keys}
    norm = _normalize_weights(weights, keys)
    raw = {str(k): float(total) * float(norm[str(k)]) for k in keys}
    alloc = {str(k): int(raw[str(k)]) for k in keys}
    remainder = int(total) - sum(alloc.values())
    if remainder > 0:
        order = sorted(keys, key=lambda k: (raw[str(k)] - float(alloc[str(k)])), reverse=True)
        idx = 0
        while remainder > 0 and order:
            key = str(order[idx % len(order)])
            alloc[key] += 1
            remainder -= 1
            idx += 1
    return alloc


def _quantize_pressure(raw: float, bins: Sequence[float]) -> int:
    b0, b1, b2 = [float(x) for x in bins]
    if raw <= b0:
        return 1
    if raw <= b1:
        return 2
    if raw <= b2:
        return 3
    return 3


def _label_with_pi0(
    *,
    projector: Pi0IntentAwareV2,
    sample_id: str,
    source: str,
    source_type: str,
    source_trust: str,
    text: str,
    is_attack: int,
    ordinal_bins: Sequence[float],
) -> Dict[str, Any]:
    text_norm = str(text).strip()
    if not text_norm:
        raise ValueError(f"empty text for sample_id={sample_id}")
    if int(is_attack) == 0:
        wall_labels = [0, 0, 0, 0]
        pressure_level = [0, 0, 0, 0]
        polarity = [0, 0, 0, 0]
    else:
        proj = projector.project(
            ContentItem(
                doc_id=str(sample_id),
                source_id=str(source),
                source_type=str(source_type),
                trust=str(source_trust),
                text=text_norm,
            )
        )
        wall_labels = [1 if float(v) > 0.0 else 0 for v in proj.v.tolist()]
        polarity = [int(x) for x in proj.evidence.polarity]
        raw_scores = [float(x) for x in proj.evidence.debug_scores_raw]
        pressure_level = [
            _quantize_pressure(raw_scores[i], ordinal_bins) if int(wall_labels[i]) == 1 else 0
            for i in range(4)
        ]
        if sum(wall_labels) == 0:
            wall_labels = [1, 0, 0, 0]
            pressure_level = [2, 0, 0, 0]
            polarity = [1, 0, 0, 0]
    for i in range(4):
        if int(wall_labels[i]) == 0:
            pressure_level[i] = 0
            polarity[i] = 0
    approx_tokens = _word_tokens(text_norm)
    return {
        "sample_id": str(sample_id),
        "text": text_norm,
        "wall_labels": [int(x) for x in wall_labels],
        "pressure_level": [int(x) for x in pressure_level],
        "polarity": [int(x) for x in polarity],
        "source": str(source),
        "source_type": str(source_type),
        "source_trust": str(source_trust),
        "lang": "en",
        "split": "holdout",
        "label_quality": "weak_prefill",
        "is_attack": int(is_attack),
        "chunk_bucket": _chunk_bucket(approx_tokens),
        "approx_tokens": int(approx_tokens),
    }


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            yield row


def _load_wainject_rows(root: str, *, limit_per_file: int = 500, seed: int = 41) -> List[Dict[str, Any]]:
    base = Path(root)
    out: List[Dict[str, Any]] = []
    rng = random.Random(seed)
    for split_name, is_attack in (("benign", 0), ("malicious", 1)):
        paths = sorted((base / split_name).glob("*.jsonl"))
        for path in paths:
            rows = []
            for row in _iter_jsonl(path):
                text = str(row.get("text", row.get("content", ""))).strip()
                if not text:
                    continue
                rows.append({"text": text, "is_attack": int(is_attack), "source_file": path.stem})
            if len(rows) > int(limit_per_file):
                idx = sorted(rng.sample(range(len(rows)), k=int(limit_per_file)))
                rows = [rows[i] for i in idx]
            out.extend(rows)
    return out


def _collect_source_candidates(cfg: PrefillConfig, projector: Pi0IntentAwareV2) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {"deepset_train": [], "wainject_text": [], "redteam_synth_train": []}
    budget = _allocate_counts(
        max(1, int(cfg.target_size) * 3),
        cfg.source_weights,
        ["deepset_train", "wainject_text", "redteam_synth_train"],
    )
    rng = random.Random(cfg.seed + 503)

    deepset_bundle = build_deepset_samples(
        cfg.deepset_root,
        split="train",
        mode="sampled",
        max_samples=max(1, int(budget.get("deepset_train", 300))),
        seed=cfg.seed,
        label_attack_value=1,
    )
    for sample in deepset_bundle.samples:
        defaults = SOURCE_DEFAULTS["deepset_train"]
        out["deepset_train"].append(
            _label_with_pi0(
                projector=projector,
                sample_id=f"goldpref:deepset:{sample.sample_id}",
                source="deepset_train",
                source_type=defaults["source_type"],
                source_trust=defaults["source_trust"],
                text=sample.text,
                is_attack=1 if sample.is_attack else 0,
                ordinal_bins=cfg.ordinal_bins,
            )
        )

    w_rows = _load_wainject_rows(cfg.wainject_root, seed=cfg.seed + 19)
    if len(w_rows) > int(budget.get("wainject_text", 240)):
        idx = sorted(rng.sample(range(len(w_rows)), k=int(budget.get("wainject_text", 240))))
        w_rows = [w_rows[i] for i in idx]
    for idx, row in enumerate(w_rows, start=1):
        defaults = SOURCE_DEFAULTS["wainject_text"]
        out["wainject_text"].append(
            _label_with_pi0(
                projector=projector,
                sample_id=f"goldpref:wainject:{idx}",
                source="wainject_text",
                source_type=defaults["source_type"],
                source_trust=defaults["source_trust"],
                text=str(row["text"]),
                is_attack=int(row["is_attack"]),
                ordinal_bins=cfg.ordinal_bins,
            )
        )

    synth_budget = max(1, int(budget.get("redteam_synth_train", 140)))
    n_per_family = max(20, int((float(synth_budget) / 3.5) + 1))
    synth = generate(seed=cfg.seed + 71, n_per_family=n_per_family)
    if len(synth) > synth_budget:
        idx = sorted(rng.sample(range(len(synth)), k=synth_budget))
        synth = [synth[i] for i in idx]
    for idx, sample in enumerate(synth, start=1):
        defaults = SOURCE_DEFAULTS["redteam_synth_train"]
        out["redteam_synth_train"].append(
            _label_with_pi0(
                projector=projector,
                sample_id=f"goldpref:redteam:{sample.id}:{idx}",
                source="redteam_synth_train",
                source_type=defaults["source_type"],
                source_trust=defaults["source_trust"],
                text=sample.text,
                is_attack=1,
                ordinal_bins=cfg.ordinal_bins,
            )
        )
    return out


def _select_from_source(
    *,
    rows: Sequence[Mapping[str, Any]],
    target: int,
    chunk_weights: Mapping[str, float],
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if target <= 0 or not rows:
        return [], [dict(r) for r in rows]
    pool_by_bucket: Dict[str, List[Dict[str, Any]]] = {"64": [], "128_256": [], "512": []}
    for row in rows:
        pool_by_bucket[str(row.get("chunk_bucket", "64"))].append(dict(row))
    rng = random.Random(seed)
    for vals in pool_by_bucket.values():
        rng.shuffle(vals)

    bucket_targets = _allocate_counts(int(target), chunk_weights, ["64", "128_256", "512"])
    selected: List[Dict[str, Any]] = []
    for bucket in ("64", "128_256", "512"):
        take_n = min(int(bucket_targets[bucket]), len(pool_by_bucket[bucket]))
        if take_n > 0:
            selected.extend(pool_by_bucket[bucket][:take_n])
            pool_by_bucket[bucket] = pool_by_bucket[bucket][take_n:]

    if len(selected) < int(target):
        remaining = []
        for bucket in ("64", "128_256", "512"):
            remaining.extend(pool_by_bucket[bucket])
        rng.shuffle(remaining)
        need = int(target) - len(selected)
        selected.extend(remaining[:need])
        remaining = remaining[need:]
    else:
        remaining = []
        for bucket in ("64", "128_256", "512"):
            remaining.extend(pool_by_bucket[bucket])

    rng.shuffle(selected)
    return selected[: int(target)], remaining


def select_prefill_candidates(
    pools: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    target_size: int,
    source_weights: Mapping[str, float],
    chunk_weights: Mapping[str, float],
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    source_keys = sorted([str(k) for k in pools.keys()])
    source_targets = _allocate_counts(int(target_size), source_weights, source_keys)
    selected: List[Dict[str, Any]] = []
    leftovers: List[Dict[str, Any]] = []
    for idx, source in enumerate(source_keys):
        part, rest = _select_from_source(
            rows=pools.get(source, []),
            target=int(source_targets.get(source, 0)),
            chunk_weights=chunk_weights,
            seed=seed + (idx * 97),
        )
        selected.extend(part)
        leftovers.extend(rest)

    rng = random.Random(seed + 701)
    if len(selected) < int(target_size):
        rng.shuffle(leftovers)
        need = int(target_size) - len(selected)
        selected.extend(leftovers[:need])
    elif len(selected) > int(target_size):
        rng.shuffle(selected)
        selected = selected[: int(target_size)]
    rng.shuffle(selected)
    selected = _rebalance_chunk_buckets(
        selected,
        target_size=int(target_size),
        chunk_weights=chunk_weights,
        seed=seed,
    )

    source_counts: Dict[str, int] = {}
    chunk_counts: Dict[str, int] = {}
    attack = 0
    benign = 0
    for row in selected:
        source = str(row.get("source", "unknown"))
        source_counts[source] = source_counts.get(source, 0) + 1
        bucket = str(row.get("chunk_bucket", "64"))
        chunk_counts[bucket] = chunk_counts.get(bucket, 0) + 1
        if int(row.get("is_attack", 0)) == 1:
            attack += 1
        else:
            benign += 1

    selection_manifest = {
        "target_size": int(target_size),
        "selected_size": int(len(selected)),
        "seed": int(seed),
        "source_targets": source_targets,
        "source_counts": dict(sorted(source_counts.items())),
        "chunk_counts": dict(sorted(chunk_counts.items())),
        "attack_count": int(attack),
        "benign_count": int(benign),
        "sample_ids_sha256": _sha256_bytes(
            "\n".join(sorted(str(r.get("sample_id", "")) for r in selected)).encode("utf-8")
        ),
    }
    return selected, selection_manifest


def build_gold_slice_prefill(
    *,
    cfg: PrefillConfig,
    projector: Pi0IntentAwareV2,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    pools = _collect_source_candidates(cfg, projector)
    selected, selection_manifest = select_prefill_candidates(
        pools=pools,
        target_size=int(cfg.target_size),
        source_weights=cfg.source_weights,
        chunk_weights=cfg.chunk_weights,
        seed=int(cfg.seed),
    )
    return selected, selection_manifest


def write_jsonl(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), ensure_ascii=True) + "\n")


def write_json(path: str, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(payload), ensure_ascii=True, indent=2), encoding="utf-8")
