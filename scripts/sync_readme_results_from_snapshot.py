from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
START_MARKER = "<!-- RESULTS_SNAPSHOT:START -->"
END_MARKER = "<!-- RESULTS_SNAPSHOT:END -->"


def _fmt(value: float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def build_results_block(snapshot: dict) -> str:
    runs = {str(item["name"]): item for item in snapshot.get("runs", [])}
    run_a = runs["frozen_run_a_core_benchmark"]
    run_b = runs["frozen_run_b_stateful_vs_baseline_d"]

    m_a = run_a["metrics"]
    m_b = run_b["metrics"]
    model = run_b.get("baseline_d_model", "gpt-5.4-mini")

    lines = [
        START_MARKER,
        "### Results Scope (Frozen, Reproducible)",
        "",
        f"- Frozen run A: `{run_a['run_id']}`",
        f"- Frozen run B: `{run_b['run_id']}`",
        "- Source of truth: `docs/public_results_snapshot.json`",
        "",
        "| Slice | Variant | attack_off_rate | benign_off_rate | Notes |",
        "|---|---|---:|---:|---|",
        (
            f"| Run A / support_compare | stateful_target | "
            f"`{_fmt(m_a['support_compare.stateful_target.attack_off_rate'])}` | "
            f"`{_fmt(m_a['support_compare.stateful_target.benign_off_rate'])}` | "
            f"`steps_to_off_median={_fmt(m_a['support_compare.stateful_target.steps_to_off_median'])}` |"
        ),
        (
            f"| Run A / attack_layer | stateful_target | "
            f"`{_fmt(m_a['attack_layer.stateful_target.attack_off_rate'])}` | "
            f"`{_fmt(m_a['attack_layer.stateful_target.benign_off_rate'])}` | "
            f"`utility_preservation=1.0` |"
        ),
        (
            f"| Run B / overall | stateful_target | "
            f"`{_fmt(m_b['overall.stateful_target.session_attack_off_rate'])}` | "
            f"`{_fmt(m_b['overall.stateful_target.session_benign_off_rate'])}` | "
            f"`stateful session metric` |"
        ),
        (
            f"| Run B / overall | baseline_d_bare_llm_detector | "
            f"`{_fmt(m_b['overall.baseline_d_bare_llm_detector.session_attack_off_rate'])}` | "
            f"`{_fmt(m_b['overall.baseline_d_bare_llm_detector.session_benign_off_rate'])}` | "
            f"`model={model}` |"
        ),
        "",
        (
            f"> Comparative baseline-D numbers are validated for `{model}` only. "
            "Equivalent behavior on other models is not claimed."
        ),
        END_MARKER,
    ]
    return "\n".join(lines) + "\n"


def sync_readme(*, snapshot_path: Path, readme_path: Path) -> None:
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    block = build_results_block(snapshot)
    text = readme_path.read_text(encoding="utf-8")

    if START_MARKER in text and END_MARKER in text:
        start = text.index(START_MARKER)
        end = text.index(END_MARKER) + len(END_MARKER)
        updated = text[:start] + block + text[end:]
    else:
        updated = text.rstrip() + "\n\n" + block

    readme_path.write_text(updated, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync README result section from public_results_snapshot.json.")
    parser.add_argument("--snapshot", default="docs/public_results_snapshot.json")
    parser.add_argument("--readme", default="README.md")
    args = parser.parse_args()

    sync_readme(snapshot_path=(ROOT / args.snapshot).resolve(), readme_path=(ROOT / args.readme).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
