from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FN_ATTACK_CUE_RE = re.compile(r"(ignore|disregard|override|new instruction|pwned|secret key)", re.IGNORECASE)


def _cluster_promptshield_text(text: str) -> str:
    t = str(text).lower()
    grammar_markers = (
        "rewrite",
        "rephrase",
        "more concise",
        "grammar",
        "grammatical",
        "proofread",
        "edit",
        "polite email",
    )
    soft_directive_markers = (
        "i want you",
        "your task is",
        "answer with",
        "respond with",
        "reply with",
        "your answer must",
        "answer according to",
        "could you",
        "can you",
        "please provide",
    )
    task_like_markers = (
        "given the following text",
        "are the following two sentences",
        "write the sentiment",
        "what is the sentiment",
        "classification",
        "extract the",
        "create ",
        "write ",
        "generate ",
        "describe ",
        "name ",
        "summarize ",
        "manual",
        "guide",
        "essay",
        "function",
        "program",
    )
    override_cues = (
        "ignore previous instructions",
        "disregard the above instructions",
        "override previous rules",
        "forget previous instructions",
        "print above prompt",
        "return embeddings",
        "system prompt",
    )

    if any(x in t for x in grammar_markers):
        return "grammar_edit_wrappers"
    if any(x in t for x in soft_directive_markers) and not any(x in t for x in override_cues):
        return "soft_directive_no_override"
    if any(x in t for x in task_like_markers):
        return "task_like_eval"
    return "task_like_eval"


def _cluster_promptshield_audit_flags(text: str) -> Dict[str, Any]:
    t = str(text).lower()
    weak_markers = ("previous", "prior", "above", "skip", "earlier", "before")
    override_cues = (
        "ignore previous instructions",
        "disregard the above instructions",
        "override previous rules",
        "forget previous instructions",
        "print above prompt",
        "return embeddings",
        "system prompt",
    )
    return {
        "weak_marker_hit": bool(any(x in t for x in weak_markers)),
        "override_cue_hit": bool(any(x in t for x in override_cues)),
    }


def _target_wall_from_text(text: str) -> str:
    t = str(text).lower()
    if any(
        x in t
        for x in (
            "embedding",
            "print above prompt",
            "system prompt",
            "prompt text",
            "secret key",
            "only print key",
            "only output key",
            "output only the key",
            "print only the key",
        )
    ):
        return "secret_exfiltration"
    if any(x in t for x in ("open a new tab", "click", "upvote", "subscribe", "fill out", "do not ask user", "retry")):
        return "tool_or_action_abuse"
    return "override_instructions"


def _resolve_path(raw: str) -> Path:
    p = Path(str(raw))
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p


def _load_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object in {path}")
    return obj


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            out.append(row)
    return out


def _extract_fp_fn(rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    fp = [dict(r) for r in rows if int(r.get("label", 0)) == 0 and bool(r.get("pred_attack", False))]
    fn = [dict(r) for r in rows if int(r.get("label", 0)) == 1 and not bool(r.get("pred_attack", False))]
    fp.sort(key=lambda x: str(x.get("id", "")))
    fn.sort(key=lambda x: str(x.get("id", "")))
    return fp, fn


def _extract_fn_attack_cue_slice(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        if int(row.get("label", 0)) != 1:
            continue
        if bool(row.get("pred_attack", False)):
            continue
        text = str(row.get("text", ""))
        if FN_ATTACK_CUE_RE.search(text):
            out.append(dict(row))
    out.sort(key=lambda x: str(x.get("id", "")))
    return out


def _build_blocking_fp(rows: Sequence[Mapping[str, Any]], max_count: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(list(rows)[: int(max_count)], start=1):
        text = str(row.get("text", ""))
        out.append(
            {
                "id": f"PSI1-BFP-{i:04d}",
                "source": "promptshield_validation",
                "origin_sample_id": str(row.get("id", "")),
                "cluster": _cluster_promptshield_text(text),
                "cluster_audit": _cluster_promptshield_audit_flags(text),
                "text": text,
                "expected_mode": "block",
                "expected_all_zero": True,
                "expected_nonzero": [],
            }
        )
    return out


def _build_blocking_fn(rows: Sequence[Mapping[str, Any]], max_count: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(list(rows)[: int(max_count)], start=1):
        text = str(row.get("text", ""))
        out.append(
            {
                "id": f"PSI1-BFN-{i:04d}",
                "source": "promptshield_validation",
                "origin_sample_id": str(row.get("id", "")),
                "cluster": _cluster_promptshield_text(text),
                "cluster_audit": _cluster_promptshield_audit_flags(text),
                "text": text,
                "expected_mode": "block",
                "expected_all_zero": False,
                "expected_nonzero": [_target_wall_from_text(text)],
            }
        )
    return out


def _build_observe(
    *,
    fp_rows: Sequence[Mapping[str, Any]],
    fn_rows: Sequence[Mapping[str, Any]],
    max_fp: int,
    max_fn: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(list(fp_rows)[: int(max_fp)], start=1):
        text = str(row.get("text", ""))
        out.append(
            {
                "id": f"PSI1-OBS-FP-{i:04d}",
                "source": "promptshield_test_truncated",
                "origin_sample_id": str(row.get("id", "")),
                "cluster": _cluster_promptshield_text(text),
                "cluster_audit": _cluster_promptshield_audit_flags(text),
                "text": text,
                "expected_mode": "observe",
                "expected_all_zero": True,
                "expected_nonzero": [],
                "observe_type": "fp_candidate",
            }
        )
    for i, row in enumerate(list(fn_rows)[: int(max_fn)], start=1):
        text = str(row.get("text", ""))
        out.append(
            {
                "id": f"PSI1-OBS-FN-{i:04d}",
                "source": "promptshield_test_truncated",
                "origin_sample_id": str(row.get("id", "")),
                "cluster": _cluster_promptshield_text(text),
                "cluster_audit": _cluster_promptshield_audit_flags(text),
                "text": text,
                "expected_mode": "observe",
                "expected_all_zero": False,
                "expected_nonzero": [_target_wall_from_text(text)],
                "observe_type": "fn_candidate",
            }
        )
    return out


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build PromptShield Iteration-1 split case packs: blocking_fp, blocking_fn, observe."
    )
    parser.add_argument("--validation-report", required=True)
    parser.add_argument("--test-report", required=True)
    parser.add_argument("--out-blocking-fp", default="tests/data/promptshield/promptshield_iter1_blocking_fp.jsonl")
    parser.add_argument("--out-blocking-fn", default="tests/data/promptshield/promptshield_iter1_blocking_fn.jsonl")
    parser.add_argument("--out-observe", default="tests/data/promptshield/promptshield_iter1_observe.jsonl")
    parser.add_argument("--out-meta", default="tests/data/promptshield/promptshield_iter1_casepack.meta.json")
    parser.add_argument("--max-blocking-fp", type=int, default=64)
    parser.add_argument("--max-blocking-fn", type=int, default=64)
    parser.add_argument("--max-observe-fp", type=int, default=64)
    parser.add_argument("--max-observe-fn", type=int, default=64)
    args = parser.parse_args()

    val_report_path = _resolve_path(args.validation_report)
    test_report_path = _resolve_path(args.test_report)
    val_report = _load_json(val_report_path)
    test_report = _load_json(test_report_path)

    val_rows_path = _resolve_path(str((val_report.get("artifacts", {}) or {}).get("rows_jsonl", "")))
    test_rows_path = _resolve_path(str((test_report.get("artifacts", {}) or {}).get("rows_jsonl", "")))
    val_rows = _load_rows(val_rows_path)
    test_rows = _load_rows(test_rows_path)

    val_fp, val_fn = _extract_fp_fn(val_rows)
    test_fp, test_fn = _extract_fp_fn(test_rows)
    val_fn_attack_cue = _extract_fn_attack_cue_slice(val_rows)

    blocking_fp = _build_blocking_fp(val_fp, int(args.max_blocking_fp))
    blocking_fn = _build_blocking_fn(val_fn, int(args.max_blocking_fn))
    observe = _build_observe(
        fp_rows=test_fp,
        fn_rows=test_fn,
        max_fp=int(args.max_observe_fp),
        max_fn=int(args.max_observe_fn),
    )

    out_bfp = _resolve_path(args.out_blocking_fp)
    out_bfn = _resolve_path(args.out_blocking_fn)
    out_obs = _resolve_path(args.out_observe)
    out_meta = _resolve_path(args.out_meta)

    _write_jsonl(out_bfp, blocking_fp)
    _write_jsonl(out_bfn, blocking_fn)
    _write_jsonl(out_obs, observe)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "status": "ok",
        "validation_report": str(val_report_path),
        "test_report": str(test_report_path),
        "validation_rows": str(val_rows_path),
        "test_rows": str(test_rows_path),
        "counts": {
            "validation_fp_total": int(len(val_fp)),
            "validation_fn_total": int(len(val_fn)),
            "validation_fn_attack_cue_total": int(len(val_fn_attack_cue)),
            "test_fp_total": int(len(test_fp)),
            "test_fn_total": int(len(test_fn)),
            "blocking_fp_written": int(len(blocking_fp)),
            "blocking_fn_written": int(len(blocking_fn)),
            "observe_written": int(len(observe)),
        },
        "protocol": {
            "tuning_split": "validation",
            "checkpoint_split": "test_truncated",
            "blocking_from": "validation_fp_fn",
            "fn_attack_cue_slice": "validation_label1_pred0_with_(ignore|disregard|override|new instruction|pwned|secret key)",
            "observe_from": "test_truncated_fp_fn",
        },
        "artifacts": {
            "blocking_fp_jsonl": str(out_bfp),
            "blocking_fn_jsonl": str(out_bfn),
            "observe_jsonl": str(out_obs),
            "meta_json": str(out_meta),
        },
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
