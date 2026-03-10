"""BIPIA dataset adapter for Omega evaluation."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


TEXT_TASKS = ("email", "table", "qa", "abstract")
CODE_TASKS = ("code",)
ALL_TASKS = TEXT_TASKS + CODE_TASKS


@dataclass(frozen=True)
class BIPIASample:
    sample_id: str
    task: str
    is_attack: bool
    text: str
    attack_name: Optional[str] = None
    position: Optional[str] = None


@dataclass(frozen=True)
class BIPIATaskBundle:
    task: str
    contexts_total: int
    contexts_selected: int
    attack_prompts_total: int
    attack_prompts_selected: int
    attack_samples: List[BIPIASample]
    benign_samples: List[BIPIASample]


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _flatten_attacks(path: Path) -> Dict[str, str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Attack JSON must be mapping: {path}")
    flat: Dict[str, str] = {}
    for attack_name, prompts in obj.items():
        if not isinstance(prompts, list):
            continue
        for idx, prompt in enumerate(prompts):
            if not isinstance(prompt, str):
                continue
            flat[f"{attack_name}-{idx}"] = prompt
    return flat


def _render_context(task: str, row: dict) -> str:
    if task in TEXT_TASKS:
        return str(row.get("context", "")).strip()
    if task in CODE_TASKS:
        context = row.get("context", "")
        if isinstance(context, list):
            return "\n".join(str(x) for x in context).strip()
        return str(context).strip()
    raise ValueError(f"Unsupported BIPIA task: {task}")


def _insert_start(context: str, attack: str) -> str:
    return f"{attack}\n{context}".strip()


def _insert_end(context: str, attack: str) -> str:
    return f"{context}\n{attack}".strip()


def _insert_middle(context: str, attack: str) -> str:
    if not context:
        return attack
    splits = list(re.finditer(r"[.!?\n]+", context))
    if not splits:
        pivot = max(1, len(context) // 2)
        return f"{context[:pivot]}\n{attack}\n{context[pivot:]}".strip()
    mid = len(splits) // 2
    pivot = splits[mid].end()
    return f"{context[:pivot]}\n{attack}\n{context[pivot:]}".strip()


def _positions() -> List[str]:
    return ["start", "middle", "end"]


def _apply_position(context: str, attack: str, position: str) -> str:
    if position == "start":
        return _insert_start(context=context, attack=attack)
    if position == "middle":
        return _insert_middle(context=context, attack=attack)
    if position == "end":
        return _insert_end(context=context, attack=attack)
    raise ValueError(f"Unsupported insertion position: {position}")


def _pick_rows(rows: List[dict], *, mode: str, max_contexts: int, seed: int) -> List[dict]:
    if mode == "full":
        return list(rows)
    if max_contexts <= 0 or len(rows) <= max_contexts:
        return list(rows)
    rng = random.Random(seed)
    indexes = sorted(rng.sample(range(len(rows)), k=max_contexts))
    return [rows[i] for i in indexes]


def _pick_attacks(attacks: Dict[str, str], *, mode: str, max_attacks: int, seed: int) -> Dict[str, str]:
    if mode == "full":
        return dict(attacks)
    if max_attacks <= 0 or len(attacks) <= max_attacks:
        return dict(attacks)
    keys = sorted(attacks.keys())
    rng = random.Random(seed)
    selected_keys = sorted(rng.sample(keys, k=max_attacks))
    return {k: attacks[k] for k in selected_keys}


def _task_attack_file(task: str, benchmark_root: Path) -> Path:
    if task in TEXT_TASKS:
        return benchmark_root / "text_attack_test.json"
    if task in CODE_TASKS:
        return benchmark_root / "code_attack_test.json"
    raise ValueError(f"Unsupported BIPIA task: {task}")


def iter_required_bipia_files(benchmark_root: str, split: str) -> Iterable[Path]:
    root = Path(benchmark_root)
    for task in ALL_TASKS:
        yield root / task / f"{split}.jsonl"
    yield root / "text_attack_test.json"
    yield root / "code_attack_test.json"
    yield root / "qa" / "md5.txt"
    yield root / "abstract" / "md5.txt"


def build_bipia_task_bundles(
    benchmark_root: str,
    *,
    split: str = "test",
    mode: str = "sampled",
    max_contexts_per_task: int = 20,
    max_attacks_per_task: int = 10,
    seed: int = 41,
) -> Dict[str, BIPIATaskBundle]:
    root = Path(benchmark_root)
    if split != "test":
        raise ValueError("BIPIA v1 adapter currently supports split=test only")
    if mode not in {"sampled", "full"}:
        raise ValueError("mode must be sampled|full")

    bundles: Dict[str, BIPIATaskBundle] = {}
    for task_idx, task in enumerate(ALL_TASKS):
        context_rows = _load_jsonl(root / task / f"{split}.jsonl")
        selected_rows = _pick_rows(
            context_rows,
            mode=mode,
            max_contexts=max_contexts_per_task,
            seed=seed + (task_idx * 37),
        )

        flat_attacks = _flatten_attacks(_task_attack_file(task, root))
        selected_attacks = _pick_attacks(
            flat_attacks,
            mode=mode,
            max_attacks=max_attacks_per_task,
            seed=seed + (task_idx * 97),
        )

        attack_samples: List[BIPIASample] = []
        benign_samples: List[BIPIASample] = []
        positions = _positions()
        for row_idx, row in enumerate(selected_rows):
            context = _render_context(task, row)
            benign_samples.append(
                BIPIASample(
                    sample_id=f"{task}:{split}:benign:{row_idx}",
                    task=task,
                    is_attack=False,
                    text=context,
                )
            )
            for attack_name in sorted(selected_attacks.keys()):
                attack = selected_attacks[attack_name]
                for position in positions:
                    attacked = _apply_position(context=context, attack=attack, position=position)
                    attack_samples.append(
                        BIPIASample(
                            sample_id=f"{task}:{split}:attack:{row_idx}:{attack_name}:{position}",
                            task=task,
                            is_attack=True,
                            text=attacked,
                            attack_name=attack_name,
                            position=position,
                        )
                    )

        bundles[task] = BIPIATaskBundle(
            task=task,
            contexts_total=len(context_rows),
            contexts_selected=len(selected_rows),
            attack_prompts_total=len(flat_attacks),
            attack_prompts_selected=len(selected_attacks),
            attack_samples=attack_samples,
            benign_samples=benign_samples,
        )
    return bundles
