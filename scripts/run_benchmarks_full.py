from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run(name: str, cmd: list[str], run_dir: Path) -> dict[str, Any]:
    start = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    elapsed = round(time.time() - start, 3)
    stdout_file = run_dir / f"{name}.stdout.txt"
    stderr_file = run_dir / f"{name}.stderr.txt"
    stdout_file.write_text(proc.stdout, encoding="utf-8")
    stderr_file.write_text(proc.stderr, encoding="utf-8")
    return {
        "name": name,
        "cmd": cmd,
        "exit_code": int(proc.returncode),
        "elapsed_sec": float(elapsed),
        "stdout_file": str(stdout_file.relative_to(ROOT).as_posix()),
        "stderr_file": str(stderr_file.relative_to(ROOT).as_posix()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full benchmark reproduction for public top-3 tracks.")
    parser.add_argument("--artifacts-root", default="artifacts/benchmarks_full")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--api-provider", default="openai")
    parser.add_argument("--api-model", default="gpt-5.4-mini")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    run_id = f"benchmarks_full_{_utc_compact()}"
    run_dir = (ROOT / str(args.artifacts_root) / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []

    pack_path = ROOT / "tests/data/session_benchmark/agentdojo_cocktail_mini_v1.jsonl"
    runs_root = ROOT / "data/AgentDojo/runs"
    wainject_root = ROOT / "data/WAInjectBench/text"
    promptshield_root = ROOT / "data/PromptShield"

    if not pack_path.exists() and not runs_root.exists():
        failures.append("AgentDojo prerequisites missing: expected tests/data/session_benchmark/agentdojo_cocktail_mini_v1.jsonl or data/AgentDojo/runs")
    if not wainject_root.exists():
        failures.append("WAInjectBench prerequisites missing: data/WAInjectBench/text")
    if not promptshield_root.exists():
        failures.append("PromptShield prerequisites missing: data/PromptShield")

    api_key_name = str(args.api_key_env)
    if not str(os.getenv(api_key_name, "")).strip():
        failures.append(f"API key env missing for full AgentDojo hybrid_api run: {api_key_name}")

    if failures:
        payload = {
            "run_id": run_id,
            "status": "failed_preconditions",
            "failures": failures,
            "artifacts_dir": str(run_dir.relative_to(ROOT).as_posix()),
        }
        (run_dir / "report.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=True, indent=2))
        return 2

    steps: list[dict[str, Any]] = []

    if not pack_path.exists() and runs_root.exists():
        steps.append(
            _run(
                "build_agentdojo_pack",
                [
                    sys.executable,
                    "scripts/build_agentdojo_cocktail_mini_pack.py",
                    "--runs-root",
                    str(runs_root.as_posix()),
                    "--out",
                    str(pack_path.as_posix()),
                    "--meta-out",
                    str((ROOT / "tests/data/session_benchmark/agentdojo_cocktail_mini_v1.meta.json").as_posix()),
                    "--seed",
                    str(int(args.seed)),
                ],
                run_dir,
            )
        )

    steps.append(
        _run(
            "eval_agentdojo_stateful_mini",
            [
                sys.executable,
                "scripts/eval_agentdojo_stateful_mini.py",
                "--profile",
                str(args.profile),
                "--mode",
                "hybrid_api",
                "--pack",
                str(pack_path.as_posix()),
                "--seed",
                str(int(args.seed)),
                "--strict-projector",
                "--blind-eval",
                "--api-provider",
                str(args.api_provider),
                "--api-model",
                str(args.api_model),
                "--api-key-env",
                str(args.api_key_env),
                "--artifacts-root",
                "artifacts/agentdojo_stateful_mini_eval",
            ],
            run_dir,
        )
    )

    steps.append(
        _run(
            "eval_wainjectbench_text",
            [
                sys.executable,
                "scripts/eval_wainjectbench_text.py",
                "--profile",
                str(args.profile),
                "--root",
                str(wainject_root.as_posix()),
                "--seed",
                str(int(args.seed)),
                "--artifacts-root",
                "artifacts/wainject_eval",
            ],
            run_dir,
        )
    )

    steps.append(
        _run(
            "eval_promptshield_text",
            [
                sys.executable,
                "scripts/eval_promptshield_text.py",
                "--profile",
                str(args.profile),
                "--root",
                str(promptshield_root.as_posix()),
                "--split",
                "validation",
                "--seed",
                str(int(args.seed)),
                "--artifacts-root",
                "artifacts/promptshield_eval",
            ],
            run_dir,
        )
    )

    failing = [step for step in steps if int(step.get("exit_code", 1)) != 0]
    payload = {
        "run_id": run_id,
        "status": "ok" if not failing else "failed_steps",
        "steps": steps,
        "failed_step_count": int(len(failing)),
        "artifacts_dir": str(run_dir.relative_to(ROOT).as_posix()),
    }
    (run_dir / "report.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return 0 if not failing else 1


if __name__ == "__main__":
    raise SystemExit(main())
