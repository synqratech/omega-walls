from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


FAMILY_COCKTAIL = "cocktail"
FAMILY_DISTRIBUTED = "distributed_wo_explicit"
FAMILY_ROLEPLAY = "roleplay_escalation"
FAMILY_TOOL = "tool_exfil_chain"
FAMILY_BENIGN = "benign_long_context"

ATTACK_FAMILIES = [FAMILY_COCKTAIL, FAMILY_DISTRIBUTED, FAMILY_ROLEPLAY, FAMILY_TOOL]
ALL_FAMILIES = ATTACK_FAMILIES + [FAMILY_BENIGN]


@dataclass(frozen=True)
class TextSample:
    text: str
    source_ref: str


def infer_eval_slice_from_source_ref(source_ref: str) -> str:
    ref = str(source_ref or "")
    if "EIA_wo_EI.jsonl" in ref or "VWA_adv_wo_EI.jsonl" in ref:
        return "context_required"
    return "text_intrinsic"


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for ln in path.read_text(encoding="utf-8").splitlines():
        line = ln.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _load_wainject_pools(repo_root: Path) -> Dict[str, List[TextSample]]:
    base = repo_root / "data" / "WAInjectBench" / "text"
    benign_files = {"comment_issue.jsonl", "email_msg.jsonl", "img_caption.jsonl", "web_text.jsonl"}
    distributed_files = {"EIA_wo_EI.jsonl", "VWA_adv_wo_EI.jsonl", "popup.jsonl", "wasp.jsonl"}
    cocktail_files = {"EIA_w_EI.jsonl", "VWA_adv_w_EI.jsonl"}
    tool_files = {"VPI_web_text.jsonl", "VPI_E_M.jsonl"}

    pools: Dict[str, List[TextSample]] = defaultdict(list)
    for split in ("benign", "malicious"):
        split_dir = base / split
        if not split_dir.exists():
            continue
        for file in sorted(split_dir.glob("*.jsonl")):
            rows = _load_jsonl(file)
            for idx, row in enumerate(rows):
                text = str(row.get("text", row.get("content", ""))).strip()
                if not text:
                    continue
                sid = row.get("id", idx + 1)
                sample = TextSample(text=text, source_ref=f"wainject:{split}:{file.name}:{sid}")
                if file.name in benign_files:
                    pools[FAMILY_BENIGN].append(sample)
                if file.name in distributed_files:
                    pools[FAMILY_DISTRIBUTED].append(sample)
                if file.name in cocktail_files:
                    pools[FAMILY_COCKTAIL].append(sample)
                if file.name in tool_files:
                    pools[FAMILY_TOOL].append(sample)
    return pools


def _load_strict_holdout_pools(repo_root: Path) -> Dict[str, List[TextSample]]:
    path = repo_root / "tests" / "data" / "strict_pi_holdout" / "strict_pi_holdout_seed41.jsonl"
    pools: Dict[str, List[TextSample]] = defaultdict(list)
    for row in _load_jsonl(path):
        label = str(row.get("label", "0")).strip().lower()
        family = str(row.get("family", "")).strip().lower()
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        rid = str(row.get("id", "unknown"))
        if label in {"1", "attack", "true"} and family == "roleplay":
            pools[FAMILY_ROLEPLAY].append(TextSample(text=text, source_ref=f"strict:{rid}"))
        if label in {"0", "benign", "false"}:
            pools[FAMILY_BENIGN].append(TextSample(text=text, source_ref=f"strict:{rid}"))
    return pools


def _load_local_contour_pools(repo_root: Path) -> Dict[str, List[TextSample]]:
    base = repo_root / "data" / "local_contour"
    pools: Dict[str, List[TextSample]] = defaultdict(list)
    if not base.exists():
        return pools

    for file in sorted((base / "trusted").glob("*.txt")):
        text = file.read_text(encoding="utf-8").strip()
        if text:
            pools[FAMILY_BENIGN].append(TextSample(text=text, source_ref=f"local_contour:{file.name}"))
    for file in sorted((base / "semi_trusted").glob("*.txt")):
        text = file.read_text(encoding="utf-8").strip()
        if text:
            pools[FAMILY_BENIGN].append(TextSample(text=text, source_ref=f"local_contour:{file.name}"))

    for file in sorted((base / "untrusted").glob("*.txt")):
        text = file.read_text(encoding="utf-8").strip()
        if not text:
            continue
        name = file.name.lower()
        ref = f"local_contour:{file.name}"
        if "distributed_" in name and "distributed_5" not in name:
            pools[FAMILY_DISTRIBUTED].append(TextSample(text=text, source_ref=ref))
        if name in {
            "distributed_5.txt",
            "malicious_doc.txt",
            "web_article_prompt_injection.txt",
            "obfuscated_attack.txt",
            "adaptive_override.txt",
            "adaptive_exfil.txt",
        }:
            pools[FAMILY_COCKTAIL].append(TextSample(text=text, source_ref=ref))
        if name in {"adaptive_tool.txt", "web_article_prompt_injection.txt", "obfuscated_attack.txt"}:
            pools[FAMILY_TOOL].append(TextSample(text=text, source_ref=ref))
    return pools


def _merge_pools(*blocks: Mapping[str, Sequence[TextSample]]) -> Dict[str, List[TextSample]]:
    out: Dict[str, List[TextSample]] = defaultdict(list)
    for block in blocks:
        for family, rows in block.items():
            out[family].extend(list(rows))
    return out


def _pick(pool: Sequence[TextSample], rng: random.Random) -> TextSample:
    if not pool:
        raise ValueError("empty pool")
    return pool[int(rng.randrange(0, len(pool)))]


def _turn_count(rng: random.Random) -> int:
    return int(rng.randint(3, 8))


def _render_session_rows(
    *,
    session_id: str,
    actor_id: str,
    bucket: str,
    family: str,
    labels: Sequence[str],
    samples: Sequence[TextSample],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    session_label = "attack" if family != FAMILY_BENIGN else "benign"
    for idx, (lab, sample) in enumerate(zip(labels, samples), start=1):
        rows.append(
            {
                "session_id": session_id,
                "turn_id": int(idx),
                "text": sample.text,
                "label_turn": str(lab),
                "label_session": session_label,
                "family": family,
                "source_ref": sample.source_ref,
                "actor_id": actor_id,
                "bucket": bucket,
                "eval_slice": infer_eval_slice_from_source_ref(sample.source_ref),
            }
        )
    return rows


def build_session_pack(*, repo_root: Path, seed: int = 41) -> List[Dict[str, Any]]:
    rng = random.Random(int(seed))
    wa = _load_wainject_pools(repo_root)
    strict = _load_strict_holdout_pools(repo_root)
    local = _load_local_contour_pools(repo_root)
    pools = _merge_pools(wa, strict, local)

    for fam in ALL_FAMILIES:
        if not pools.get(fam):
            raise RuntimeError(f"missing data pool for family={fam}")

    attack_quotas = {
        FAMILY_COCKTAIL: 54,
        FAMILY_DISTRIBUTED: 54,
        FAMILY_ROLEPLAY: 51,
        FAMILY_TOOL: 51,
    }
    attack_total = int(sum(attack_quotas.values()))
    if attack_total != 210:
        raise RuntimeError("attack quota must be 210")

    attack_specs: List[Tuple[str, bool]] = []
    for family, quota in attack_quotas.items():
        attack_specs.extend([(family, False) for _ in range(int(quota))])
    rng.shuffle(attack_specs)

    mixed_total = int(round(0.20 * attack_total))
    if mixed_total != 42:
        raise RuntimeError("mixed attack quota must be 42")
    attack_specs = [
        (family, idx < mixed_total) for idx, (family, _) in enumerate(attack_specs)
    ]

    need_cross = {FAMILY_COCKTAIL: 15, FAMILY_DISTRIBUTED: 15}
    cross_indices: List[int] = []
    for idx, (family, _) in enumerate(attack_specs):
        if family in need_cross and need_cross[family] > 0:
            cross_indices.append(idx)
            need_cross[family] -= 1
        if sum(need_cross.values()) <= 0:
            break
    if len(cross_indices) != 30:
        raise RuntimeError("cross-session attack quota must be 30")
    cross_index_set = set(cross_indices)

    rows: List[Dict[str, Any]] = []
    cross_actor_ids = [f"actor-cross-{i:02d}" for i in range(10)]
    cross_actor_pointer = 0

    for idx, (family, is_mixed) in enumerate(attack_specs, start=1):
        is_cross = (idx - 1) in cross_index_set
        bucket = "cross_session" if is_cross else "core"
        if is_cross:
            actor_id = cross_actor_ids[(cross_actor_pointer // 3) % len(cross_actor_ids)]
            cross_actor_pointer += 1
            session_id = f"sess-xs-atk-{idx:03d}"
        else:
            actor_id = f"actor-core-atk-{idx:03d}"
            session_id = f"sess-core-atk-{idx:03d}"

        n_turns = _turn_count(rng)
        labels: List[str]
        samples: List[TextSample]
        if is_mixed:
            attack_turns = max(1, n_turns - 2)
            labels = ["benign"] + ["attack"] * attack_turns + ["benign"]
            samples = [_pick(pools[FAMILY_BENIGN], rng)]
            samples.extend(_pick(pools[family], rng) for _ in range(attack_turns))
            samples.append(_pick(pools[FAMILY_BENIGN], rng))
        else:
            labels = ["attack"] * n_turns
            samples = [_pick(pools[family], rng) for _ in range(n_turns)]
        rows.extend(
            _render_session_rows(
                session_id=session_id,
                actor_id=actor_id,
                bucket=bucket,
                family=family,
                labels=labels,
                samples=samples,
            )
        )

    for idx in range(1, 91):
        session_id = f"sess-core-ben-{idx:03d}"
        actor_id = f"actor-core-ben-{idx:03d}"
        n_turns = _turn_count(rng)
        labels = ["benign"] * n_turns
        samples = [_pick(pools[FAMILY_BENIGN], rng) for _ in range(n_turns)]
        rows.extend(
            _render_session_rows(
                session_id=session_id,
                actor_id=actor_id,
                bucket="core",
                family=FAMILY_BENIGN,
                labels=labels,
                samples=samples,
            )
        )

    rows = sorted(rows, key=lambda r: (str(r["session_id"]), int(r["turn_id"])))
    return rows


def _session_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["session_id"])].append(row)
    for sid in grouped:
        grouped[sid] = sorted(grouped[sid], key=lambda r: int(r["turn_id"]))
    return grouped


def _summary(rows: Sequence[Mapping[str, Any]], seed: int) -> Dict[str, Any]:
    grouped = _session_rows(rows)
    sessions = list(grouped.values())
    attack_sessions = [s for s in sessions if str(s[0]["label_session"]) == "attack"]
    benign_sessions = [s for s in sessions if str(s[0]["label_session"]) == "benign"]
    mixed_attack = 0
    session_len_counter: Counter[int] = Counter()
    family_counter: Counter[str] = Counter()
    bucket_counter: Counter[str] = Counter()
    eval_slice_counter_rows: Counter[str] = Counter()
    eval_slice_counter_sessions: Counter[str] = Counter()
    for ses in sessions:
        first = ses[0]
        family_counter[str(first["family"])] += 1
        bucket_counter[str(first["bucket"])] += 1
        session_len_counter[len(ses)] += 1
        ses_slice = "text_intrinsic"
        for row in ses:
            row_slice = str(row.get("eval_slice", infer_eval_slice_from_source_ref(str(row.get("source_ref", "")))))
            eval_slice_counter_rows[row_slice] += 1
            if row_slice == "context_required":
                ses_slice = "context_required"
        eval_slice_counter_sessions[ses_slice] += 1
        if str(first["label_session"]) == "attack":
            turns = {str(r["label_turn"]) for r in ses}
            if "attack" in turns and "benign" in turns:
                mixed_attack += 1
    return {
        "seed": int(seed),
        "rows_total": int(len(rows)),
        "sessions_total": int(len(sessions)),
        "attack_sessions": int(len(attack_sessions)),
        "benign_sessions": int(len(benign_sessions)),
        "mixed_attack_sessions": int(mixed_attack),
        "by_family_sessions": dict(sorted(family_counter.items())),
        "by_bucket_sessions": dict(sorted(bucket_counter.items())),
        "by_eval_slice_rows": dict(sorted(eval_slice_counter_rows.items())),
        "by_eval_slice_sessions": dict(sorted(eval_slice_counter_sessions.items())),
        "session_length_histogram": {str(k): int(v) for k, v in sorted(session_len_counter.items())},
    }


def write_session_pack(rows: Sequence[Mapping[str, Any]], *, out_jsonl: Path, seed: int) -> Dict[str, Any]:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    summary = _summary(rows, seed=seed)
    meta = out_jsonl.with_suffix(".meta.json")
    meta.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "jsonl": str(out_jsonl),
        "meta_json": str(meta),
        "summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build fixed session-based PI benchmark pack (seeded deterministic).")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--out", default="tests/data/session_benchmark/session_pack_seed41_v1.jsonl")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    rows = build_session_pack(repo_root=root, seed=int(args.seed))
    result = write_session_pack(rows, out_jsonl=(root / str(args.out)).resolve(), seed=int(args.seed))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
