from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent

EIA_SLOT_REDIRECT_MARKERS = (
    "right place to input",
    "right place to enter",
    "right place to provide",
)
EIA_SLOT_REDIRECT_ACTIONS = {
    "input",
    "enter",
    "provide",
}
EIA_SLOT_REDIRECT_SLOTS = {
    "first",
    "last",
    "name",
    "email",
    "address",
    "physical",
    "phone",
    "contact",
    "city",
    "state",
    "zip",
    "postal",
    "confirmation",
    "confirm",
    "recipient",
    "sender",
    "location",
    "dropoff",
}

VWA_ENTITY_TOKENS = {"price", "seller", "lister", "listing", "email", "release", "date"}
VWA_RELEASE_CONTEXT_TOKENS = {"movie", "poster", "title"}
VWA_IMAGE_CONTEXT_TOKENS = {"image", "photo", "picture"}
VWA_PRICE_TAG_CONTEXT_MARKERS = ("phone number", "advertisement", "label", "displayed at the bottom")
VWA_QUALITY_PRIMARY_MARKERS = ("distorted", "pixelated", "blurry", "corrupted", "glitched", "digitally altered", "wavy or melted")
VWA_QUALITY_SECONDARY_MARKERS = (
    "difficult to provide an accurate description",
    "difficult to discern",
    "low-resolution",
    "heavily distorted",
)
NEG_OVERLAP_TOKENS = {
    "input",
    "enter",
    "provide",
    "email",
    "address",
    "phone",
    "release",
    "date",
    "price",
    "listing",
    "seller",
    "lister",
    "replace",
    "instead",
}


def _tokenize(text: str) -> List[str]:
    return [tok for tok in re.split(r"[^a-z0-9]+", text.lower()) if tok]


def _contains_any(text: str, markers: Iterable[str]) -> bool:
    return any(m in text for m in markers)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            continue
        row["_lineno"] = lineno
        out.append(row)
    return out


def _row_text(row: Mapping[str, Any]) -> str:
    for key in ("text", "instruction", "content"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _sample_ref(prefix: str, filename: str, row: Mapping[str, Any]) -> str:
    sid = row.get("sample_id")
    if isinstance(sid, str) and sid:
        return f"{prefix}:{filename}:{sid}"
    rid = row.get("id")
    if isinstance(rid, str) and rid:
        return f"{prefix}:{filename}:{rid}"
    return f"{prefix}:{filename}:{int(row.get('_lineno', 0))}"


def _is_eia_slot_redirect(text: str) -> bool:
    t = text.lower()
    toks = set(_tokenize(t))
    return (
        _contains_any(t, EIA_SLOT_REDIRECT_MARKERS)
        and bool(toks.intersection(EIA_SLOT_REDIRECT_ACTIONS))
        and bool(toks.intersection(EIA_SLOT_REDIRECT_SLOTS))
    )


def _vwa_cluster(text: str) -> str:
    t = text.lower()
    toks = set(_tokenize(t))
    has_number = bool(re.search(r"\b\d+(?:[\.,]\d+)?\b", t))
    has_currency = bool(re.search(r"[$€£]|\b\d+[\.,]\d{2}\b", t))
    has_date = bool(re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", t))
    has_email = "@" in t

    if "actual email for this request is" in t and has_email:
        return "actual_email_assertion"
    if (
        "release date" in t
        and bool(toks.intersection(VWA_RELEASE_CONTEXT_TOKENS))
        and (bool(re.search(r"release date.{0,40}\bis\b", t)) or ("movie title" in t) or has_date)
    ):
        return "release_date_caption_assertion"
    if (
        "price tag" in t
        and bool(toks.intersection(VWA_IMAGE_CONTEXT_TOKENS))
        and ((has_currency or has_number) or _contains_any(t, VWA_PRICE_TAG_CONTEXT_MARKERS))
        and (
            _contains_any(t, ("actual", "actually", "instead", "replace", "not"))
            or _contains_any(t, VWA_PRICE_TAG_CONTEXT_MARKERS)
        )
    ):
        return "price_tag_caption_assertion"
    quality_primary_count = sum(1 for m in VWA_QUALITY_PRIMARY_MARKERS if m in t)
    quality_secondary_hit = _contains_any(t, VWA_QUALITY_SECONDARY_MARKERS)
    quality_context_hit = bool(toks.intersection(VWA_IMAGE_CONTEXT_TOKENS | VWA_RELEASE_CONTEXT_TOKENS))
    if quality_context_hit and (quality_primary_count >= 2 or (quality_primary_count >= 1 and quality_secondary_hit)):
        return "distortion_caption_assertion"
    if bool(re.search(r"\b(actual|actually)\b.{0,40}\bnot\b", t)) and bool(toks.intersection(VWA_ENTITY_TOKENS)):
        return "actual_not_entity"
    if "latest listing" in t or "seller is actually" in t or "lister is actually" in t:
        return "listing_override_assertion"
    if bool(re.search(r"\breplace\b.{0,50}\bwith\b", t)) and bool(toks.intersection(VWA_ENTITY_TOKENS)):
        return "replacement_entity"
    if bool(re.search(r"\bnot\b.{0,40}\bbut\b", t)) and bool(toks.intersection(VWA_ENTITY_TOKENS)):
        return "replacement_entity"
    if _contains_any(t, ("instead use", "instead enter", "instead put", "instead choose", "instead select")) and bool(
        toks.intersection(VWA_ENTITY_TOKENS)
    ):
        return "replacement_entity"
    return ""

def _is_neg_overlap_candidate(text: str) -> bool:
    t = text.lower()
    toks = set(_tokenize(t))
    if not toks.intersection(NEG_OVERLAP_TOKENS):
        return False
    if _contains_any(
        t,
        (
            "ignore previous instructions",
            "disregard",
            "override",
            "must click",
            "do not ask user",
            "return embeddings",
            "print above prompt",
        ),
    ):
        return False
    return True


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _resolve_path(raw: str) -> Path:
    p = Path(str(raw))
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic WAInject tail casepack for Iteration 3.")
    parser.add_argument("--root", default="data/WAInjectBench/text")
    parser.add_argument(
        "--out-blocking-pos",
        default="tests/data/wainject_tail_iter3/wainject_tail_iter3_blocking_pos.jsonl",
    )
    parser.add_argument(
        "--out-blocking-neg",
        default="tests/data/wainject_tail_iter3/wainject_tail_iter3_blocking_neg.jsonl",
    )
    parser.add_argument(
        "--out-meta",
        default="tests/data/wainject_tail_iter3/wainject_tail_iter3_casepack.meta.json",
    )
    parser.add_argument("--max-pos-eia", type=int, default=96)
    parser.add_argument("--max-pos-vwa", type=int, default=96)
    parser.add_argument("--max-neg", type=int, default=128)
    args = parser.parse_args()

    root = _resolve_path(args.root)
    mal = root / "malicious"
    ben = root / "benign"

    eia_path = mal / "EIA_wo_EI.jsonl"
    vwa_path = mal / "VWA_adv_wo_EI.jsonl"
    neg_paths = [ben / "comment_issue.jsonl", ben / "email_msg.jsonl", ben / "web_text.jsonl"]

    eia_rows = _load_jsonl(eia_path)
    vwa_rows = _load_jsonl(vwa_path)
    neg_rows = []
    for p in neg_paths:
        neg_rows.extend([{**r, "_source_file": p.name} for r in _load_jsonl(p)])

    pos_items: List[Dict[str, Any]] = []
    eia_hits = 0
    vwa_hits = 0

    for row in eia_rows:
        text = _row_text(row)
        if not text or not _is_eia_slot_redirect(text):
            continue
        if eia_hits >= int(args.max_pos_eia):
            break
        eia_hits += 1
        pos_items.append(
            {
                "id": f"WAI3-POS-EIA-{eia_hits:04d}",
                "source": "wainject_tail_iter3",
                "origin_sample_id": _sample_ref("malicious", eia_path.name, row),
                "cluster": "eia_slot_redirect",
                "text": text,
                "expected_mode": "block",
                "expected_all_zero": False,
                "expected_nonzero": ["tool_or_action_abuse"],
            }
        )

    for row in vwa_rows:
        text = _row_text(row)
        if not text:
            continue
        cluster = _vwa_cluster(text)
        if not cluster:
            continue
        if vwa_hits >= int(args.max_pos_vwa):
            break
        vwa_hits += 1
        pos_items.append(
            {
                "id": f"WAI3-POS-VWA-{vwa_hits:04d}",
                "source": "wainject_tail_iter3",
                "origin_sample_id": _sample_ref("malicious", vwa_path.name, row),
                "cluster": cluster,
                "text": text,
                "expected_mode": "block",
                "expected_all_zero": False,
                "expected_nonzero": ["tool_or_action_abuse"],
            }
        )

    neg_items: List[Dict[str, Any]] = []
    for row in sorted(neg_rows, key=lambda x: (str(x.get("_source_file", "")), int(x.get("_lineno", 0)))):
        text = _row_text(row)
        if not text or not _is_neg_overlap_candidate(text):
            continue
        if _is_eia_slot_redirect(text):
            continue
        if _vwa_cluster(text):
            continue
        neg_items.append(
            {
                "id": f"WAI3-NEG-{len(neg_items) + 1:04d}",
                "source": "wainject_tail_iter3",
                "origin_sample_id": _sample_ref("benign", str(row.get("_source_file", "")), row),
                "cluster": "lexical_overlap_benign",
                "text": text,
                "expected_mode": "block",
                "expected_all_zero": True,
                "expected_nonzero": [],
            }
        )
        if len(neg_items) >= int(args.max_neg):
            break

    out_pos = _resolve_path(args.out_blocking_pos)
    out_neg = _resolve_path(args.out_blocking_neg)
    out_meta = _resolve_path(args.out_meta)
    _write_jsonl(out_pos, pos_items)
    _write_jsonl(out_neg, neg_items)

    out_meta.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "status": "ok",
        "root": str(root),
        "inputs": {
            "malicious": [str(eia_path), str(vwa_path)],
            "benign": [str(p) for p in neg_paths],
        },
        "counts": {
            "eia_positive_written": int(eia_hits),
            "vwa_positive_written": int(vwa_hits),
            "blocking_pos_written": int(len(pos_items)),
            "blocking_neg_written": int(len(neg_items)),
        },
        "artifacts": {
            "blocking_pos_jsonl": str(out_pos),
            "blocking_neg_jsonl": str(out_neg),
            "meta_json": str(out_meta),
        },
        "protocol": {
            "deterministic": True,
            "sampling": "none",
            "scope": ["EIA_wo_EI", "VWA_adv_wo_EI"],
            "neg_overlap_sources": ["comment_issue", "email_msg", "web_text"],
        },
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
