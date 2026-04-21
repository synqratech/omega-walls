from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build_replay_argv(
    *,
    profile: str,
    replay_input: str,
    state_bootstrap: str,
    deterministic_runs: int,
    strict: bool,
) -> list[str]:
    argv = [
        sys.executable,
        "scripts/replay_incident.py",
        "--replay-input",
        replay_input,
        "--profile",
        profile,
        "--state-bootstrap",
        state_bootstrap,
        "--deterministic-runs",
        str(int(deterministic_runs)),
    ]
    if strict:
        argv.append("--strict")
    return argv


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict deterministic replay smoke wrapper.")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--replay-input", default="tests/data/replay/replay_strict_smoke_input.json")
    parser.add_argument(
        "--state-bootstrap",
        choices=["fresh_state", "reuse_state", "reset_actor_before_run"],
        default="fresh_state",
    )
    parser.add_argument("--deterministic-runs", type=int, default=2)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    replay_path = (ROOT / str(args.replay_input)).resolve()
    if not replay_path.exists():
        raise FileNotFoundError(f"Replay input not found: {replay_path}")

    argv = build_replay_argv(
        profile=str(args.profile),
        replay_input=str(replay_path.as_posix()),
        state_bootstrap=str(args.state_bootstrap),
        deterministic_runs=max(2, int(args.deterministic_runs)),
        strict=bool(args.strict),
    )
    proc = subprocess.run(argv, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)

    payload = {
        "event": "replay_strict_smoke_v1",
        "argv": argv,
        "exit_code": int(proc.returncode),
        "status": "ok" if int(proc.returncode) == 0 else "fail",
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
