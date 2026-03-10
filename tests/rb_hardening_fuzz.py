from __future__ import annotations

import base64
import random
import re
import urllib.parse
from typing import Callable

from tests.rb_hardening_suite_cases import get_base_rb_hardening_cases

ZERO_WIDTH = "\u200b"


def _replace_casefold(text: str, pattern: str, repl: str) -> str:
    return re.sub(re.escape(pattern), repl, text, flags=re.IGNORECASE)


def op_spacing_tokens(text: str) -> str:
    out = text
    for token in ("ignore", "reveal", "token", "bypass", "run", "command", "prompt"):
        out = _replace_casefold(out, token, " ".join(list(token)))
    return out


def op_leet_swap(text: str) -> str:
    mapping = str.maketrans(
        {
            "a": "4",
            "e": "3",
            "i": "1",
            "o": "0",
            "s": "5",
            "t": "7",
            "A": "4",
            "E": "3",
            "I": "1",
            "O": "0",
            "S": "5",
            "T": "7",
        }
    )
    return text.translate(mapping)


def op_code_fence_wrap(text: str) -> str:
    return f"```text\n{text}\n```"


def op_url_lite_wrap(text: str) -> str:
    encoded = urllib.parse.quote(text, safe="")
    # Keep original phrase visible to avoid turning this op into full encoding.
    return f"url-lite://q={encoded}\n{text}"


def op_zero_width_insert(text: str) -> str:
    out = text
    for token in ("ignore", "reveal", "token", "bypass", "prompt"):
        broken = ZERO_WIDTH.join(list(token))
        out = _replace_casefold(out, token, broken)
    return out


def op_base64_lite_wrap(text: str) -> str:
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return f"[base64-lite]{encoded}[/base64-lite]"


def _build_mutated_case(base: dict, *, suffix: str, op_name: str, text: str, expected_mode: str) -> dict:
    return {
        "id": f"{base['id']}__{suffix}_{op_name}",
        "group": str(base["group"]),
        "targets": list(base["targets"]),
        "text": text,
        "expected_mode": expected_mode,
        "note": f"{base.get('note', '')}; mutation={op_name}",
    }


def build_rb_hardening_cases(seed: int = 41) -> list[dict]:
    rng = random.Random(int(seed))
    base_cases = get_base_rb_hardening_cases()
    cases = [dict(row) for row in base_cases]

    block_ops: list[tuple[str, Callable[[str], str]]] = [
        ("spacing_tokens", op_spacing_tokens),
        ("leet_swap", op_leet_swap),
        ("code_fence_wrap", op_code_fence_wrap),
        ("url_lite_wrap", op_url_lite_wrap),
    ]
    observe_ops: list[tuple[str, Callable[[str], str]]] = [
        ("zero_width_insert", op_zero_width_insert),
        ("base64_lite_wrap", op_base64_lite_wrap),
    ]

    # One blocking mutation per base case: +40
    for idx, base in enumerate(base_cases):
        block_idx = (int(seed) + idx) % len(block_ops)
        op_name, op_fn = block_ops[block_idx]
        if str(base["group"]) != "positive" and op_name in {"code_fence_wrap", "url_lite_wrap"}:
            op_name, op_fn = ("leet_swap", op_leet_swap)
        mutated = _build_mutated_case(
            base,
            suffix="mut_block",
            op_name=op_name,
            text=op_fn(str(base["text"])),
            expected_mode="block",
        )
        cases.append(mutated)

    # Observe-only mutation for half base cases: +20
    observe_indices = list(range(0, len(base_cases), 2))
    rng.shuffle(observe_indices)
    observe_indices = sorted(observe_indices[:20])
    for pos, idx in enumerate(observe_indices):
        base = base_cases[idx]
        op_idx = (int(seed) + pos) % len(observe_ops)
        op_name, op_fn = observe_ops[op_idx]
        mutated = _build_mutated_case(
            base,
            suffix="mut_observe",
            op_name=op_name,
            text=op_fn(str(base["text"])),
            expected_mode="observe",
        )
        cases.append(mutated)

    if len(cases) != 100:
        raise ValueError(f"rb_hardening case budget mismatch: expected 100, got {len(cases)}")
    seen = set()
    for row in cases:
        rid = str(row["id"])
        if rid in seen:
            raise ValueError(f"duplicate case id: {rid}")
        seen.add(rid)
    return cases
