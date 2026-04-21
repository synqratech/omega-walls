from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.config.loader import load_resolved_config
from omega.core.omega_core import OmegaCoreV1
from omega.core.params import omega_params_from_config
from omega.interfaces.contracts_v1 import ContentItem, OmegaState, WALLS_V1
from omega.policy.off_policy_v1 import OffPolicyV1
from omega.projector.factory import build_projector


WAINJECT_SOURCE_URL = "https://github.com/Norrrrrrr-lyn/WAInjectBench"
WAINJECT_PAPER_URL = "https://arxiv.org/abs/2510.01354"
WAINJECT_SOURCE_DATE_UTC = "2026-03-08"


@dataclass(frozen=True)
class WATextSample:
    sample_id: str
    label: int
    source_file: str
    text: str


@dataclass(frozen=True)
class WATextLoadStats:
    files_seen: int
    benign_files_seen: int
    malicious_files_seen: int
    rows_seen: int
    rows_loaded: int
    rows_dropped_empty_line: int
    rows_dropped_invalid_json: int
    rows_dropped_non_mapping: int
    rows_dropped_empty_text: int
    samples_built: int
    benign_samples_built: int
    malicious_samples_built: int


@dataclass(frozen=True)
class SessionizedTurn:
    session_id: str
    turn_id: int
    source_file: str
    label_turn: int
    text: str
    label_session: int


def _utc_compact_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if float(b) != 0.0 else 0.0


def _metrics(tp: int, fp: int, tn: int, fn: int) -> Dict[str, Any]:
    return {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "attack_total": int(tp + fn),
        "benign_total": int(tn + fp),
        "attack_off_rate": _safe_div(tp, tp + fn),
        "benign_off_rate": _safe_div(fp, tn + fp),
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
    }


def _load_jsonl_rows(path: Path) -> tuple[List[Mapping[str, Any]], Dict[str, int]]:
    out: List[Mapping[str, Any]] = []
    stats = {
        "rows_seen": 0,
        "rows_loaded": 0,
        "rows_dropped_empty_line": 0,
        "rows_dropped_invalid_json": 0,
        "rows_dropped_non_mapping": 0,
    }
    for raw in path.read_text(encoding="utf-8").splitlines():
        stats["rows_seen"] += 1
        line = raw.strip()
        if not line:
            stats["rows_dropped_empty_line"] += 1
            continue
        try:
            obj = json.loads(line)
        except Exception:
            stats["rows_dropped_invalid_json"] += 1
            continue
        if isinstance(obj, Mapping):
            out.append(obj)
            stats["rows_loaded"] += 1
        else:
            stats["rows_dropped_non_mapping"] += 1
    return out, stats


def load_wainject_text_with_stats(root: Path) -> tuple[List[WATextSample], WATextLoadStats]:
    samples: List[WATextSample] = []
    files_seen = 0
    benign_files_seen = 0
    malicious_files_seen = 0
    rows_seen = 0
    rows_loaded = 0
    rows_dropped_empty_line = 0
    rows_dropped_invalid_json = 0
    rows_dropped_non_mapping = 0
    rows_dropped_empty_text = 0
    benign_samples_built = 0
    malicious_samples_built = 0

    for split_dir, label in (("benign", 0), ("malicious", 1)):
        base = root / split_dir
        if not base.exists():
            continue
        for file in sorted(base.glob("*.jsonl")):
            files_seen += 1
            if split_dir == "benign":
                benign_files_seen += 1
            else:
                malicious_files_seen += 1
            rows, row_stats = _load_jsonl_rows(file)
            rows_seen += int(row_stats["rows_seen"])
            rows_loaded += int(row_stats["rows_loaded"])
            rows_dropped_empty_line += int(row_stats["rows_dropped_empty_line"])
            rows_dropped_invalid_json += int(row_stats["rows_dropped_invalid_json"])
            rows_dropped_non_mapping += int(row_stats["rows_dropped_non_mapping"])
            for idx, row in enumerate(rows):
                text = str(row.get("text", row.get("content", ""))).strip()
                if not text:
                    rows_dropped_empty_text += 1
                    continue
                samples.append(
                    WATextSample(
                        sample_id=f"{split_dir}:{file.name}:{idx:06d}",
                        label=int(label),
                        source_file=file.name,
                        text=text,
                    )
                )
                if split_dir == "benign":
                    benign_samples_built += 1
                else:
                    malicious_samples_built += 1
    stats = WATextLoadStats(
        files_seen=files_seen,
        benign_files_seen=benign_files_seen,
        malicious_files_seen=malicious_files_seen,
        rows_seen=rows_seen,
        rows_loaded=rows_loaded,
        rows_dropped_empty_line=rows_dropped_empty_line,
        rows_dropped_invalid_json=rows_dropped_invalid_json,
        rows_dropped_non_mapping=rows_dropped_non_mapping,
        rows_dropped_empty_text=rows_dropped_empty_text,
        samples_built=len(samples),
        benign_samples_built=benign_samples_built,
        malicious_samples_built=malicious_samples_built,
    )
    return samples, stats


def load_wainject_text(root: Path) -> List[WATextSample]:
    samples, _ = load_wainject_text_with_stats(root)
    return samples


def _stratified_cap_samples(
    samples: Sequence[WATextSample],
    *,
    max_samples: int,
    seed: int,
) -> tuple[List[WATextSample], Dict[str, Any]]:
    if max_samples <= 0 or len(samples) <= max_samples:
        return list(samples), {"strategy": "full", "requested_max_samples": int(max_samples), "selected_total": int(len(samples))}

    grouped: Dict[tuple[int, str], List[WATextSample]] = {}
    for s in samples:
        key = (int(s.label), str(s.source_file))
        grouped.setdefault(key, []).append(s)

    rng = np.random.RandomState(int(seed))
    keys = sorted(grouped.keys(), key=lambda x: (x[0], x[1]))
    total = float(len(samples))

    floors: Dict[tuple[int, str], int] = {}
    remainders: List[tuple[float, tuple[int, str]]] = []
    for k in keys:
        group = grouped[k]
        # deterministic order within stratum
        perm = rng.permutation(len(group))
        grouped[k] = [group[int(i)] for i in perm]
        quota_raw = float(max_samples) * float(len(group)) / total
        quota_floor = min(len(group), int(np.floor(quota_raw)))
        floors[k] = quota_floor
        remainders.append((quota_raw - float(quota_floor), k))

    used = int(sum(floors.values()))
    remain = int(max_samples - used)
    for _, k in sorted(remainders, key=lambda x: (-x[0], x[1])):  # largest remainder first
        if remain <= 0:
            break
        if floors[k] < len(grouped[k]):
            floors[k] += 1
            remain -= 1

    selected: List[WATextSample] = []
    by_stratum: Dict[str, Dict[str, int]] = {}
    for k in keys:
        picked = grouped[k][: floors[k]]
        selected.extend(picked)
        lbl, fname = k
        by_stratum[f"{lbl}:{fname}"] = {"available": len(grouped[k]), "selected": len(picked)}
    selected.sort(key=lambda s: s.sample_id)
    return selected, {
        "strategy": "stratified_label_source_file",
        "requested_max_samples": int(max_samples),
        "selected_total": int(len(selected)),
        "by_stratum": by_stratum,
    }


def _chunk_rows(rows: Sequence[WATextSample], size: int) -> List[List[WATextSample]]:
    if size <= 0:
        size = 1
    out: List[List[WATextSample]] = []
    cur: List[WATextSample] = []
    for row in rows:
        cur.append(row)
        if len(cur) >= size:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def _build_sessionized_turns(
    *,
    rows: Sequence[WATextSample],
    seed: int,
    attack_chunk_size: int,
    benign_chunk_size: int,
    benign_prefix_turns: int,
) -> tuple[List[SessionizedTurn], Dict[str, Any]]:
    rng = np.random.RandomState(int(seed))
    benign_rows = [r for r in rows if int(r.label) == 0]
    attack_rows = [r for r in rows if int(r.label) == 1]

    benign_by_file: Dict[str, List[WATextSample]] = {}
    attack_by_file: Dict[str, List[WATextSample]] = {}
    for r in benign_rows:
        benign_by_file.setdefault(r.source_file, []).append(r)
    for r in attack_rows:
        attack_by_file.setdefault(r.source_file, []).append(r)

    for bucket in (benign_by_file, attack_by_file):
        for key in list(bucket.keys()):
            vals = bucket[key]
            perm = rng.permutation(len(vals))
            bucket[key] = [vals[int(i)] for i in perm]

    benign_pool = list(benign_rows)
    if benign_pool:
        perm = rng.permutation(len(benign_pool))
        benign_pool = [benign_pool[int(i)] for i in perm]
    benign_ptr = 0

    turns: List[SessionizedTurn] = []
    session_idx = 0
    per_session_meta: Dict[str, Dict[str, Any]] = {}

    # Benign sessions: benign-only chunks.
    for src in sorted(benign_by_file.keys()):
        for chunk in _chunk_rows(benign_by_file[src], benign_chunk_size):
            session_idx += 1
            session_id = f"wa-sess-ben-{session_idx:05d}"
            per_session_meta[session_id] = {"label_session": 0, "source_file": src, "turns": len(chunk), "prefixed_benign_turns": 0}
            for idx, row in enumerate(chunk, start=1):
                turns.append(
                    SessionizedTurn(
                        session_id=session_id,
                        turn_id=idx,
                        source_file=row.source_file,
                        label_turn=0,
                        text=row.text,
                        label_session=0,
                    )
                )

    # Attack sessions: optional benign prefix + malicious chunk.
    for src in sorted(attack_by_file.keys()):
        for chunk in _chunk_rows(attack_by_file[src], attack_chunk_size):
            session_idx += 1
            session_id = f"wa-sess-atk-{session_idx:05d}"
            turns_local: List[SessionizedTurn] = []
            added_prefix = 0
            if benign_pool and benign_prefix_turns > 0:
                for _ in range(benign_prefix_turns):
                    b = benign_pool[benign_ptr % len(benign_pool)]
                    benign_ptr += 1
                    added_prefix += 1
                    turns_local.append(
                        SessionizedTurn(
                            session_id=session_id,
                            turn_id=len(turns_local) + 1,
                            source_file=b.source_file,
                            label_turn=0,
                            text=b.text,
                            label_session=1,
                        )
                    )
            for row in chunk:
                turns_local.append(
                    SessionizedTurn(
                        session_id=session_id,
                        turn_id=len(turns_local) + 1,
                        source_file=row.source_file,
                        label_turn=1,
                        text=row.text,
                        label_session=1,
                    )
                )
            turns.extend(turns_local)
            per_session_meta[session_id] = {
                "label_session": 1,
                "source_file": src,
                "turns": len(turns_local),
                "prefixed_benign_turns": int(added_prefix),
            }

    turns.sort(key=lambda t: (t.session_id, int(t.turn_id)))
    diag_info = {
        "strategy": "synthetic_sessionization_from_text_rows",
        "seed": int(seed),
        "attack_chunk_size": int(attack_chunk_size),
        "benign_chunk_size": int(benign_chunk_size),
        "benign_prefix_turns": int(benign_prefix_turns),
        "sessions_total": int(len(per_session_meta)),
        "sessions_attack": int(sum(1 for x in per_session_meta.values() if int(x["label_session"]) == 1)),
        "sessions_benign": int(sum(1 for x in per_session_meta.values() if int(x["label_session"]) == 0)),
    }
    return turns, diag_info


def evaluate_wainject_sessionized_diagnostic(
    *,
    rows: Sequence[WATextSample],
    projector: Any,
    omega_core: OmegaCoreV1,
    off_policy: OffPolicyV1,
    seed: int,
    attack_chunk_size: int,
    benign_chunk_size: int,
    benign_prefix_turns: int,
) -> Dict[str, Any]:
    turns, diag_info = _build_sessionized_turns(
        rows=rows,
        seed=seed,
        attack_chunk_size=attack_chunk_size,
        benign_chunk_size=benign_chunk_size,
        benign_prefix_turns=benign_prefix_turns,
    )
    if not turns:
        return {
            "status": "dataset_not_ready",
            "comparability_status": "non_comparable",
            "note": "sessionized diagnostic is synthetic and not benchmark-comparable",
            "diagnostic_info": diag_info,
            "summary": _metrics(0, 0, 0, 0),
            "time_to_off": {"count_detected": 0, "median": None, "p95": None},
            "late_detect_rate": 0.0,
            "first_off_turn_histogram": {},
            "never_detected_rate_by_source_file": {},
        }

    sessions: Dict[str, List[SessionizedTurn]] = {}
    for t in turns:
        sessions.setdefault(t.session_id, []).append(t)
    for sid in sessions:
        sessions[sid].sort(key=lambda x: int(x.turn_id))

    session_tp = session_fp = session_tn = session_fn = 0
    detected_turns: List[int] = []
    late_count = 0
    first_hist: Dict[str, int] = {}
    misses_by_source: Dict[str, Dict[str, int]] = {}

    rows_out: List[Dict[str, Any]] = []
    for sid in sorted(sessions.keys()):
        turns_local = sessions[sid]
        label_session = int(turns_local[0].label_session)
        state = OmegaState(session_id=f"wainject_sessionized:{sid}", m=np.zeros(len(WALLS_V1), dtype=float), step=0)
        session_off = False
        first_off_turn: Optional[int] = None
        source_for_session = str(next((t.source_file for t in turns_local if int(t.label_turn) == 1), turns_local[0].source_file))

        for t in turns_local:
            item = ContentItem(
                doc_id=f"wainject_sessionized:{sid}:{int(t.turn_id):03d}",
                source_id="wainject_text_sessionized",
                source_type="other",
                trust="untrusted",
                text=t.text,
            )
            proj = projector.project(item)
            step_result = omega_core.step(state=state, items=[item], projections=[proj])
            decision = off_policy.select_actions(step_result=step_result, items=[item])
            pred_turn_off = bool(decision.off)
            state = OmegaState(
                session_id=str(step_result.session_id),
                m=np.array(step_result.m_next, dtype=float),
                step=int(step_result.step),
            )

            rows_out.append(
                {
                    "session_id": sid,
                    "turn_id": int(t.turn_id),
                    "label_turn": int(t.label_turn),
                    "label_session": int(label_session),
                    "source_file": str(t.source_file),
                    "off": pred_turn_off,
                    "severity": str(decision.severity),
                    "max_p": float(np.max(step_result.p)) if step_result.p is not None and np.size(step_result.p) > 0 else 0.0,
                }
            )
            if pred_turn_off and first_off_turn is None:
                first_off_turn = int(t.turn_id)
                session_off = True

        if label_session == 1 and session_off:
            session_tp += 1
        elif label_session == 1 and not session_off:
            session_fn += 1
        elif label_session == 0 and session_off:
            session_fp += 1
        else:
            session_tn += 1

        if label_session == 1:
            misses_by_source.setdefault(source_for_session, {"attack_total": 0, "detected": 0, "never_detected": 0})
            misses_by_source[source_for_session]["attack_total"] += 1
            if session_off:
                misses_by_source[source_for_session]["detected"] += 1
            else:
                misses_by_source[source_for_session]["never_detected"] += 1

        if first_off_turn is None:
            first_hist["never"] = int(first_hist.get("never", 0) + 1)
        else:
            detected_turns.append(int(first_off_turn))
            key = str(int(first_off_turn))
            first_hist[key] = int(first_hist.get(key, 0) + 1)
            late_thr = max(1, int(math.floor(0.7 * float(len(turns_local)))))
            if int(first_off_turn) > late_thr:
                late_count += 1

    summary = _metrics(session_tp, session_fp, session_tn, session_fn)
    late_detect_rate = _safe_div(late_count, summary["attack_total"])
    p95 = None
    if detected_turns:
        p95 = float(np.percentile(np.array(detected_turns, dtype=float), 95))

    never_by_source: Dict[str, Dict[str, Any]] = {}
    for k in sorted(misses_by_source.keys()):
        src = misses_by_source[k]
        att_total = int(src["attack_total"])
        det = int(src["detected"])
        never = int(src["never_detected"])
        never_by_source[k] = {
            "attack_total": att_total,
            "detected": det,
            "never_detected": never,
            "never_detected_rate": _safe_div(never, att_total),
            "attack_off_rate": _safe_div(det, att_total),
        }

    return {
        "status": "ok",
        "comparability_status": "non_comparable",
        "note": "Synthetic sessionization for diagnostic use only; do not compare to WAInject benchmark leaderboard.",
        "diagnostic_info": diag_info,
        "summary": summary,
        "time_to_off": {
            "count_detected": int(len(detected_turns)),
            "median": float(np.median(np.array(detected_turns, dtype=float))) if detected_turns else None,
            "p95": p95,
        },
        "late_detect_rate": float(late_detect_rate),
        "first_off_turn_histogram": first_hist,
        "never_detected_rate_by_source_file": never_by_source,
        "rows": rows_out,
    }


def evaluate_wainject_rows(
    *,
    rows: Sequence[WATextSample],
    projector: Any,
    omega_core: OmegaCoreV1,
    off_policy: OffPolicyV1,
) -> Dict[str, Any]:
    tp = fp = tn = fn = 0
    per_file_counts: Dict[str, Dict[str, int]] = {}
    eval_rows: List[Dict[str, Any]] = []

    for row in rows:
        item = ContentItem(
            doc_id=f"wainject:{row.sample_id}",
            source_id="wainject_text",
            source_type="other",
            trust="untrusted",
            text=row.text,
        )
        proj = projector.project(item)
        state = OmegaState(session_id=f"wainject:{row.sample_id}", m=np.zeros(len(WALLS_V1), dtype=float), step=0)
        step_result = omega_core.step(state=state, items=[item], projections=[proj])
        decision = off_policy.select_actions(step_result=step_result, items=[item])
        pred = bool(decision.off)

        if row.source_file not in per_file_counts:
            per_file_counts[row.source_file] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        cur = per_file_counts[row.source_file]

        if row.label == 1 and pred:
            tp += 1
            cur["tp"] += 1
        elif row.label == 1 and not pred:
            fn += 1
            cur["fn"] += 1
        elif row.label == 0 and pred:
            fp += 1
            cur["fp"] += 1
        else:
            tn += 1
            cur["tn"] += 1

        eval_rows.append(
            {
                "id": row.sample_id,
                "label": int(row.label),
                "source_file": row.source_file,
                "pred_attack": pred,
                "off": bool(decision.off),
                "severity": str(decision.severity),
                "v_sum": float(np.sum(proj.v)),
            }
        )

    per_file = {k: _metrics(v["tp"], v["fp"], v["tn"], v["fn"]) for k, v in sorted(per_file_counts.items())}
    return {"summary": _metrics(tp, fp, tn, fn), "per_file": per_file, "rows": eval_rows}


def _latest_baseline_report(artifacts_root: Path) -> Optional[Path]:
    if not artifacts_root.exists():
        return None
    candidates = sorted(
        [p / "report.json" for p in artifacts_root.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
        reverse=True,
    )
    for p in candidates:
        if not p.exists():
            continue
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, Mapping):
            continue
        status = str(payload.get("comparability_status", "")).strip().lower()
        reason = str(payload.get("comparability_reason", "")).strip().lower()
        sampling = payload.get("sampling", {})
        if not isinstance(sampling, Mapping):
            sampling = {}
        requested_max_samples = int(sampling.get("requested_max_samples", 0))
        strategy = str(sampling.get("strategy", "")).strip().lower()
        selected_total = int(sampling.get("selected_total", 0))
        samples_total = int(payload.get("samples_total", 0))
        is_full_selection = selected_total > 0 and selected_total == samples_total
        if (
            status == "partial_comparison"
            and reason == "full_run_complete_benign_malicious_splits"
            and requested_max_samples == 0
            and strategy == "full"
            and is_full_selection
        ):
            return p
    return None


def _baseline_compare(current: Mapping[str, Any], baseline: Mapping[str, Any]) -> Dict[str, Any]:
    keys = ("attack_off_rate", "benign_off_rate", "precision", "recall")
    cur = current.get("summary", {}) if isinstance(current.get("summary"), Mapping) else {}
    base = baseline.get("summary", {}) if isinstance(baseline.get("summary"), Mapping) else {}
    delta = {k: float(cur.get(k, 0.0)) - float(base.get(k, 0.0)) for k in keys}
    return {"summary_delta": delta}


def _write_external_refs_json(path: Path, *, comparability_status: str) -> None:
    payload = {
        "benchmark": "WAInjectBench",
        "source_date": WAINJECT_SOURCE_DATE_UTC,
        "sources": [
            {
                "url": WAINJECT_SOURCE_URL,
                "type": "benchmark_repo",
                "note": "official dataset repository",
            },
            {
                "url": WAINJECT_PAPER_URL,
                "type": "paper",
                "note": "official benchmark paper",
            },
        ],
        "metric_mapping": {
            "omega_metrics": ["summary.attack_off_rate", "summary.benign_off_rate", "summary.precision", "summary.recall"],
            "external_baseline_table_available": False,
            "reason": "no benchmark-maintainer leaderboard table with detector scores in public source card/readme",
        },
        "comparability_status": str(comparability_status),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_comparability_for_wainject(
    *,
    dataset_ready: bool,
    max_samples: int,
    selected_total: int,
    load_stats: WATextLoadStats,
) -> tuple[str, str]:
    if not bool(dataset_ready):
        return "non_comparable", "dataset_not_ready"
    if int(load_stats.benign_files_seen) <= 0 or int(load_stats.malicious_files_seen) <= 0:
        return "non_comparable", "missing_benign_or_malicious_split"
    if int(load_stats.benign_samples_built) <= 0 or int(load_stats.malicious_samples_built) <= 0:
        return "non_comparable", "missing_benign_or_malicious_samples"
    if int(max_samples) > 0:
        return "non_comparable", "subsampled_run_max_samples"
    if int(selected_total) != int(load_stats.samples_built):
        return "non_comparable", "selected_subset_not_full_dataset"
    return "partial_comparison", "full_run_complete_benign_malicious_splits"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Omega on WAInjectBench text subset and emit comparability metadata.")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--root", default="data/WAInjectBench/text")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap for deterministic sampled evaluation; 0 means full set.")
    parser.add_argument("--sessionized-diagnostic", action="store_true", help="Emit additional synthetic sessionized diagnostics (non-comparable).")
    parser.add_argument("--sessionized-attack-chunk-size", type=int, default=4)
    parser.add_argument("--sessionized-benign-chunk-size", type=int, default=4)
    parser.add_argument("--sessionized-benign-prefix-turns", type=int, default=1)
    parser.add_argument("--artifacts-root", default="artifacts/wainject_eval")
    parser.add_argument("--weekly-regression", action="store_true")
    parser.add_argument("--baseline-report", default=None)
    args = parser.parse_args()

    np.random.seed(int(args.seed))
    root = (ROOT / str(args.root)).resolve()
    samples, load_stats = load_wainject_text_with_stats(root)
    sampling_info: Dict[str, Any] = {"strategy": "full", "requested_max_samples": int(args.max_samples), "selected_total": int(len(samples))}
    if int(args.max_samples) > 0 and len(samples) > int(args.max_samples):
        samples, sampling_info = _stratified_cap_samples(samples, max_samples=int(args.max_samples), seed=int(args.seed))
    dataset_ready = len(samples) > 0
    comparability_status, comparability_reason = _resolve_comparability_for_wainject(
        dataset_ready=dataset_ready,
        max_samples=int(args.max_samples),
        selected_total=int(len(samples)),
        load_stats=load_stats,
    )

    snapshot = load_resolved_config(profile=args.profile)
    cfg = snapshot.resolved

    eval_block: Dict[str, Any] = {"summary": _metrics(0, 0, 0, 0), "per_file": {}, "rows": []}
    session_diag_block: Optional[Dict[str, Any]] = None
    if dataset_ready:
        projector = build_projector(cfg)
        omega_core = OmegaCoreV1(omega_params_from_config(cfg))
        off_policy = OffPolicyV1(cfg)
        eval_block = evaluate_wainject_rows(rows=samples, projector=projector, omega_core=omega_core, off_policy=off_policy)
        if bool(args.sessionized_diagnostic):
            session_diag_block = evaluate_wainject_sessionized_diagnostic(
                rows=samples,
                projector=projector,
                omega_core=omega_core,
                off_policy=off_policy,
                seed=int(args.seed),
                attack_chunk_size=int(args.sessionized_attack_chunk_size),
                benign_chunk_size=int(args.sessionized_benign_chunk_size),
                benign_prefix_turns=int(args.sessionized_benign_prefix_turns),
            )
    elif bool(args.sessionized_diagnostic):
        session_diag_block = {
            "status": "dataset_not_ready",
            "comparability_status": "non_comparable",
            "note": "sessionized diagnostic skipped because dataset is not ready",
        }

    now = datetime.now(timezone.utc)
    weekly_tag = f"_w{now.strftime('%G%V')}" if bool(args.weekly_regression) else ""
    run_id = f"wainject_eval{weekly_tag}_{_utc_compact_now()}"
    out_dir = (ROOT / str(args.artifacts_root) / run_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    rows_path = out_dir / "rows.jsonl"
    sessionized_rows_path = out_dir / "sessionized_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as fh:
        for row in eval_block["rows"]:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    session_diag_report: Optional[Dict[str, Any]] = session_diag_block
    if session_diag_block and isinstance(session_diag_block.get("rows"), list):
        with sessionized_rows_path.open("w", encoding="utf-8") as fh:
            for row in session_diag_block["rows"]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        session_diag_report = dict(session_diag_block)
        row_count = len(session_diag_block["rows"])
        session_diag_report.pop("rows", None)
        session_diag_report["rows_count"] = int(row_count)

    refs_global = ROOT / "artifacts" / "external_refs" / "wainjectbench_refs.json"
    refs_run = out_dir / "wainjectbench_refs.json"
    _write_external_refs_json(refs_global, comparability_status=comparability_status)
    _write_external_refs_json(refs_run, comparability_status=comparability_status)

    baseline_compare = None
    baseline_path = None
    if args.baseline_report:
        baseline_path = (ROOT / str(args.baseline_report)).resolve()
    elif bool(args.weekly_regression):
        latest = _latest_baseline_report((ROOT / str(args.artifacts_root)).resolve())
        if latest is not None and latest.parent.name != run_id:
            baseline_path = latest
    if baseline_path and baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_compare = {"path": str(baseline_path), **_baseline_compare(eval_block, baseline)}

    payload = {
        "run_id": run_id,
        "status": "ok" if dataset_ready else "dataset_not_ready",
        "profile": args.profile,
        "seed": int(args.seed),
        "root": str(root),
        "dataset_ready": dataset_ready,
        "samples_total": int(len(samples)),
        "dataset_rows": {
            "files_seen": int(load_stats.files_seen),
            "rows_seen": int(load_stats.rows_seen),
            "rows_loaded": int(load_stats.rows_loaded),
            "rows_dropped_empty_line": int(load_stats.rows_dropped_empty_line),
            "rows_dropped_invalid_json": int(load_stats.rows_dropped_invalid_json),
            "rows_dropped_non_mapping": int(load_stats.rows_dropped_non_mapping),
            "rows_dropped_empty_text": int(load_stats.rows_dropped_empty_text),
            "samples_built": int(load_stats.samples_built),
        },
        "sampling": sampling_info,
        "comparability_reason": str(comparability_reason),
        "evaluation_mode": {
            "runtime": "stateless_per_sample",
            "session_runtime": False,
            "note": "WAInjectBench text files are evaluated as independent samples; not a session-memory benchmark.",
        },
        "summary": eval_block["summary"],
        "per_file": eval_block["per_file"],
        "comparability_status": comparability_status,
        "external_benchmark": {
            "name": "WAInjectBench (text subset)",
            "source_url": WAINJECT_SOURCE_URL,
            "paper_url": WAINJECT_PAPER_URL,
            "source_date": WAINJECT_SOURCE_DATE_UTC,
            "metric_mapping": {
                "omega_metrics": ["summary.attack_off_rate", "summary.benign_off_rate", "summary.precision", "summary.recall"],
                "external_baseline_table_available": False,
            },
            "comparability_note": (
                "No benchmark-maintainer detector leaderboard table found in official source card/readme; reported as partial only. "
                "Comparability gate passed: full run with complete benign+malicious splits."
                if comparability_status == "partial_comparison"
                else f"Non-comparable run: {comparability_reason}."
            ),
            "evidence_bar": "benchmark-maintainer only",
        },
        "baseline_compare": baseline_compare,
        "sessionized_diagnostic": session_diag_report,
        "artifacts": {
            "report_json": str(report_path),
            "rows_jsonl": str(rows_path),
            "sessionized_rows_jsonl": str(sessionized_rows_path) if sessionized_rows_path.exists() else None,
            "external_refs_global_json": str(refs_global),
            "external_refs_run_json": str(refs_run),
        },
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
