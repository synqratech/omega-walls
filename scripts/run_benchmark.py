from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.config.loader import load_resolved_config


VARIANT_STATEFUL = "stateful_target"
VARIANT_BASELINE_D = "baseline_d_bare_llm_detector"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_MISSING_KEY = 2


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_compact_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _extract_json_blob(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("stdout does not contain JSON object")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("JSON payload is not an object")
    return payload


def _jsonl_stats(path: Path) -> Dict[str, Any]:
    row_count = 0
    family_counts: Dict[str, int] = {}
    label_session_counts: Dict[str, int] = {}
    mode_counts: Dict[str, int] = {}
    expected_off_counts: Dict[str, int] = {}

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = str(raw).strip()
        if not line:
            continue
        row_count += 1
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, Mapping):
            continue
        family = str(obj.get("family", "")).strip()
        if family:
            family_counts[family] = family_counts.get(family, 0) + 1
        label_session = str(obj.get("label_session", "")).strip()
        if label_session:
            label_session_counts[label_session] = label_session_counts.get(label_session, 0) + 1
        mode = str(obj.get("mode", "")).strip()
        if mode:
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        if "expected_off" in obj:
            key = "true" if bool(obj.get("expected_off")) else "false"
            expected_off_counts[key] = expected_off_counts.get(key, 0) + 1

    return {
        "row_count": int(row_count),
        "family_counts": dict(sorted(family_counts.items())),
        "label_session_counts": dict(sorted(label_session_counts.items())),
        "mode_counts": dict(sorted(mode_counts.items())),
        "expected_off_counts": dict(sorted(expected_off_counts.items())),
    }


def _json_stats(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, Mapping):
        top_keys = sorted(str(k) for k in obj.keys())
        counts = obj.get("counts", {})
        counts_dict = dict(counts) if isinstance(counts, Mapping) else {}
        return {"top_level_keys": top_keys, "counts": counts_dict}
    if isinstance(obj, list):
        return {"list_items": int(len(obj))}
    return {"json_type": str(type(obj).__name__)}


def _dataset_stats(path: Path) -> Dict[str, Any]:
    suffix = str(path.suffix).lower()
    if suffix == ".jsonl":
        return _jsonl_stats(path)
    if suffix == ".json":
        return _json_stats(path)
    return {"bytes": int(path.stat().st_size)}


def _load_dataset_registry(path: Path) -> Dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("benchmark dataset registry must be a mapping")
    payload = raw.get("benchmark_datasets", raw)
    if not isinstance(payload, Mapping):
        raise ValueError("benchmark_datasets key is missing or invalid")
    profiles = payload.get("profiles", {})
    if not isinstance(profiles, Mapping):
        raise ValueError("benchmark_datasets.profiles must be a mapping")
    return {"profiles": dict(profiles)}


def build_dataset_manifest(
    *,
    registry_path: Path,
    dataset_profile: str,
    seed: int,
    runtime_mode: str,
    config_snapshot_sha256: str,
) -> Dict[str, Any]:
    registry = _load_dataset_registry(registry_path)
    profiles = registry["profiles"]
    profile_payload = profiles.get(dataset_profile)
    if not isinstance(profile_payload, Mapping):
        raise ValueError(f"unknown dataset profile: {dataset_profile}")
    datasets_payload = profile_payload.get("datasets", [])
    if not isinstance(datasets_payload, list) or not datasets_payload:
        raise ValueError(f"dataset profile has no datasets: {dataset_profile}")

    datasets: List[Dict[str, Any]] = []
    for row in datasets_payload:
        if not isinstance(row, Mapping):
            raise ValueError("dataset row must be an object")
        dataset_id = str(row.get("dataset_id", "")).strip()
        rel_path = str(row.get("path", "")).strip()
        source_type = str(row.get("source_type", "")).strip()
        source_url = str(row.get("source_url", "")).strip()
        if not dataset_id or not rel_path or not source_type or not source_url:
            raise ValueError(f"incomplete dataset row: {row}")
        abs_path = (ROOT / rel_path).resolve()
        if not abs_path.exists() or not abs_path.is_file():
            raise FileNotFoundError(f"dataset file missing: {rel_path}")
        stats = _dataset_stats(abs_path)
        record = {
            "dataset_id": dataset_id,
            "path": str(abs_path),
            "path_rel": rel_path,
            "source_type": source_type,
            "source_url": source_url,
            "sha256": _sha256_file(abs_path),
            "bytes": int(abs_path.stat().st_size),
            "stats": stats,
        }
        if "row_count" in stats and int(stats["row_count"]) <= 0:
            raise ValueError(f"empty dataset rows for {dataset_id}")
        datasets.append(record)

    manifest = {
        "schema_version": "benchmark_dataset_manifest_v1",
        "generated_at_utc": _utc_now_iso(),
        "dataset_profile": dataset_profile,
        "seed": int(seed),
        "runtime_mode": str(runtime_mode),
        "config_snapshot_sha256": str(config_snapshot_sha256),
        "dataset_count": int(len(datasets)),
        "datasets": datasets,
    }
    return manifest


def _prepare_support_eval_packs(
    *,
    run_dir: Path,
    dataset_manifest: Mapping[str, Any],
) -> Path:
    root = run_dir / "support_eval_packs"
    root.mkdir(parents=True, exist_ok=True)
    datasets = dataset_manifest.get("datasets", [])
    if not isinstance(datasets, list):
        raise ValueError("dataset manifest malformed: datasets is not a list")

    selected: List[Tuple[str, Path]] = []
    for row in datasets:
        if not isinstance(row, Mapping):
            continue
        dsid = str(row.get("dataset_id", "")).strip()
        if dsid in {"session_pack_seed41_v1", "session_pack_stateful_focus_v1"}:
            selected.append((dsid, Path(str(row.get("path", ""))).resolve()))
    if not selected:
        raise ValueError("session packs are missing in dataset manifest")

    index_rows: List[Dict[str, Any]] = []
    for dsid, source_path in selected:
        pack_id = f"core_{dsid}"
        pack_root = root / pack_id
        runtime_dir = pack_root / "runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        target_pack = runtime_dir / "session_pack.jsonl"
        shutil.copyfile(source_path, target_pack)
        line_count = 0
        session_ids: set[str] = set()
        for raw in target_pack.read_text(encoding="utf-8").splitlines():
            line = str(raw).strip()
            if not line:
                continue
            line_count += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, Mapping):
                sid = str(obj.get("session_id", "")).strip()
                if sid:
                    session_ids.add(sid)
        manifest_payload = {
            "pack_id": pack_id,
            "source_dataset_id": dsid,
            "source_path": str(source_path),
            "rows": int(line_count),
            "sessions": int(len(session_ids)),
        }
        (pack_root / "manifest.json").write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (pack_root / "README.md").write_text(
            "Generated by scripts/run_benchmark.py for core_oss_v1 support compare.\n",
            encoding="utf-8",
        )
        index_rows.append(
            {
                "pack_id": pack_id,
                "pack_root": str(pack_root.resolve()),
                "runtime_pack_path": str(target_pack.resolve()),
                "manifest_path": str((pack_root / "manifest.json").resolve()),
                "readme_path": str((pack_root / "README.md").resolve()),
                "stats": {"sessions": int(len(session_ids)), "turns": int(line_count)},
            }
        )
    (root / "index.json").write_text(
        json.dumps({"packs": index_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return root


def _run_suite_command(name: str, argv: Sequence[str], out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    proc = subprocess.run(
        list(argv),
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    duration = round(time.time() - start, 3)
    stdout_file = out_dir / f"{name}.stdout.txt"
    stderr_file = out_dir / f"{name}.stderr.txt"
    stdout_file.write_text(proc.stdout, encoding="utf-8")
    stderr_file.write_text(proc.stderr, encoding="utf-8")
    parsed: Optional[Dict[str, Any]] = None
    parse_error: Optional[str] = None
    try:
        parsed = _extract_json_blob(proc.stdout)
    except Exception as exc:  # noqa: BLE001
        parse_error = str(exc)
    return {
        "name": name,
        "argv": list(argv),
        "exit_code": int(proc.returncode),
        "duration_sec": float(duration),
        "stdout_file": str(stdout_file.resolve()),
        "stderr_file": str(stderr_file.resolve()),
        "parse_error": parse_error,
        "report": parsed,
    }


def _normalize_run_eval(report: Mapping[str, Any]) -> Dict[str, Any]:
    hard = report.get("hard_negatives", {})
    hard = dict(hard) if isinstance(hard, Mapping) else {}
    can = report.get("canonical_positives", {})
    can = dict(can) if isinstance(can, Mapping) else {}
    per_wall = can.get("per_wall", {})
    per_wall = dict(per_wall) if isinstance(per_wall, Mapping) else {}
    return {
        VARIANT_STATEFUL: {
            "attack_off_rate": _safe_float(can.get("overall_hit", 0.0)),
            "hard_neg_fp": _safe_int(hard.get("fp", 0)),
            "per_wall_hit": {str(k): _safe_float(v) for k, v in sorted(per_wall.items())},
        }
    }


def _normalize_attack_layer(report: Mapping[str, Any]) -> Dict[str, Any]:
    summary = report.get("summary", {})
    summary = dict(summary) if isinstance(summary, Mapping) else {}
    return {
        VARIANT_STATEFUL: {
            "attack_off_rate": _safe_float(summary.get("attack_off_rate", 0.0)),
            "benign_off_rate": _safe_float(summary.get("benign_off_rate", 0.0)),
            "utility_preservation": _safe_float(summary.get("utility_preservation", 0.0)),
        }
    }


def _normalize_support_compare(report: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = report.get("metrics", {})
    metrics = dict(metrics) if isinstance(metrics, Mapping) else {}
    overall = metrics.get("overall", {})
    overall = dict(overall) if isinstance(overall, Mapping) else {}
    market_ready = report.get("market_ready", {})
    market_ready = dict(market_ready) if isinstance(market_ready, Mapping) else {}
    market_variants = market_ready.get("variants", {})
    market_variants = dict(market_variants) if isinstance(market_variants, Mapping) else {}

    out: Dict[str, Any] = {}
    for variant, payload in overall.items():
        if not isinstance(payload, Mapping):
            continue
        steps = payload.get("steps_to_off", {})
        steps = dict(steps) if isinstance(steps, Mapping) else {}
        enforcement_scope = "comparable"
        mrow = market_variants.get(str(variant), {})
        if isinstance(mrow, Mapping):
            scope_val = str(mrow.get("enforcement_scope", "")).strip()
            if scope_val == "detector_only_not_comparable":
                enforcement_scope = "detector_only_not_comparable"
        out[str(variant)] = {
            "attack_off_rate": _safe_float(payload.get("session_attack_off_rate", 0.0)),
            "benign_off_rate": _safe_float(payload.get("session_benign_off_rate", 0.0)),
            "late_detect_rate": _safe_float(payload.get("late_detect_rate", 0.0)),
            "steps_to_off_median": _safe_float(steps.get("median", 0.0)),
            "steps_to_off_p90": _safe_float(steps.get("p90", 0.0)),
            "utility_preservation": 1.0 - _safe_float(payload.get("session_benign_off_rate", 0.0)),
            "comparable_scope": enforcement_scope,
        }
    return out


def _headline_compare(metrics_by_variant: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    lhs = metrics_by_variant.get(VARIANT_STATEFUL)
    rhs = metrics_by_variant.get(VARIANT_BASELINE_D)
    if not isinstance(lhs, Mapping) or not isinstance(rhs, Mapping):
        return {
            "available": False,
            "lhs_variant": VARIANT_STATEFUL,
            "rhs_variant": VARIANT_BASELINE_D,
            "reason": "missing_stateful_or_baseline_d_metrics",
        }
    delta_attack = _safe_float(lhs.get("attack_off_rate", 0.0)) - _safe_float(rhs.get("attack_off_rate", 0.0))
    delta_benign = _safe_float(lhs.get("benign_off_rate", 0.0)) - _safe_float(rhs.get("benign_off_rate", 0.0))
    delta_steps = _safe_float(lhs.get("steps_to_off_median", 0.0)) - _safe_float(rhs.get("steps_to_off_median", 0.0))
    return {
        "available": True,
        "lhs_variant": VARIANT_STATEFUL,
        "rhs_variant": VARIANT_BASELINE_D,
        "metrics": {
            "lhs_attack_off_rate": _safe_float(lhs.get("attack_off_rate", 0.0)),
            "rhs_attack_off_rate": _safe_float(rhs.get("attack_off_rate", 0.0)),
            "delta_attack_off_rate": float(delta_attack),
            "lhs_benign_off_rate": _safe_float(lhs.get("benign_off_rate", 0.0)),
            "rhs_benign_off_rate": _safe_float(rhs.get("benign_off_rate", 0.0)),
            "delta_benign_off_rate": float(delta_benign),
            "lhs_steps_to_off_median": _safe_float(lhs.get("steps_to_off_median", 0.0)),
            "rhs_steps_to_off_median": _safe_float(rhs.get("steps_to_off_median", 0.0)),
            "delta_steps_to_off_median": float(delta_steps),
        },
    }


def _scorecard_row(
    *,
    run_id: str,
    suite: str,
    variant: str,
    metric: str,
    value: Any,
    unit: str,
    direction: str,
    threshold: str,
    status: str,
    comparable_scope: str,
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "suite": suite,
        "variant": variant,
        "metric": metric,
        "value": value,
        "unit": unit,
        "direction": direction,
        "threshold": threshold,
        "status": status,
        "comparable_scope": comparable_scope,
    }


def _build_scorecard_rows(
    *,
    run_id: str,
    run_eval_metrics: Mapping[str, Mapping[str, Any]],
    attack_metrics: Mapping[str, Mapping[str, Any]],
    support_metrics: Mapping[str, Mapping[str, Any]],
    headline: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for variant, payload in run_eval_metrics.items():
        rows.append(
            _scorecard_row(
                run_id=run_id,
                suite="run_eval",
                variant=variant,
                metric="hard_neg_fp",
                value=_safe_int(payload.get("hard_neg_fp", 0)),
                unit="count",
                direction="lower_is_better",
                threshold="0",
                status="ok",
                comparable_scope="comparable",
            )
        )
        rows.append(
            _scorecard_row(
                run_id=run_id,
                suite="run_eval",
                variant=variant,
                metric="attack_off_rate",
                value=round(_safe_float(payload.get("attack_off_rate", 0.0)), 6),
                unit="rate",
                direction="higher_is_better",
                threshold="",
                status="ok",
                comparable_scope="comparable",
            )
        )
        per_wall = payload.get("per_wall_hit", {})
        if isinstance(per_wall, Mapping):
            for wall, score in sorted(per_wall.items()):
                rows.append(
                    _scorecard_row(
                        run_id=run_id,
                        suite="run_eval",
                        variant=variant,
                        metric=f"per_wall_hit:{wall}",
                        value=round(_safe_float(score, 0.0), 6),
                        unit="rate",
                        direction="higher_is_better",
                        threshold="",
                        status="ok",
                        comparable_scope="comparable",
                    )
                )

    for variant, payload in attack_metrics.items():
        rows.append(
            _scorecard_row(
                run_id=run_id,
                suite="attack_layer",
                variant=variant,
                metric="attack_off_rate",
                value=round(_safe_float(payload.get("attack_off_rate", 0.0)), 6),
                unit="rate",
                direction="higher_is_better",
                threshold="",
                status="ok",
                comparable_scope="comparable",
            )
        )
        rows.append(
            _scorecard_row(
                run_id=run_id,
                suite="attack_layer",
                variant=variant,
                metric="benign_off_rate",
                value=round(_safe_float(payload.get("benign_off_rate", 0.0)), 6),
                unit="rate",
                direction="lower_is_better",
                threshold="",
                status="ok",
                comparable_scope="comparable",
            )
        )
        rows.append(
            _scorecard_row(
                run_id=run_id,
                suite="attack_layer",
                variant=variant,
                metric="utility_preservation",
                value=round(_safe_float(payload.get("utility_preservation", 0.0)), 6),
                unit="rate",
                direction="higher_is_better",
                threshold="",
                status="ok",
                comparable_scope="comparable",
            )
        )

    for variant, payload in sorted(support_metrics.items()):
        scope = str(payload.get("comparable_scope", "comparable"))
        rows.append(
            _scorecard_row(
                run_id=run_id,
                suite="support_compare",
                variant=str(variant),
                metric="attack_off_rate",
                value=round(_safe_float(payload.get("attack_off_rate", 0.0)), 6),
                unit="rate",
                direction="higher_is_better",
                threshold="",
                status="ok",
                comparable_scope=scope,
            )
        )
        rows.append(
            _scorecard_row(
                run_id=run_id,
                suite="support_compare",
                variant=str(variant),
                metric="benign_off_rate",
                value=round(_safe_float(payload.get("benign_off_rate", 0.0)), 6),
                unit="rate",
                direction="lower_is_better",
                threshold="",
                status="ok",
                comparable_scope=scope,
            )
        )
        rows.append(
            _scorecard_row(
                run_id=run_id,
                suite="support_compare",
                variant=str(variant),
                metric="steps_to_off_median",
                value=round(_safe_float(payload.get("steps_to_off_median", 0.0)), 6),
                unit="turns",
                direction="lower_is_better",
                threshold="",
                status="ok",
                comparable_scope=scope,
            )
        )

    if bool(headline.get("available")):
        hmetrics = headline.get("metrics", {})
        hmetrics = dict(hmetrics) if isinstance(hmetrics, Mapping) else {}
        rows.append(
            _scorecard_row(
                run_id=run_id,
                suite="headline_compare",
                variant=f"{VARIANT_STATEFUL}_vs_{VARIANT_BASELINE_D}",
                metric="delta_attack_off_rate",
                value=round(_safe_float(hmetrics.get("delta_attack_off_rate", 0.0)), 6),
                unit="delta_rate",
                direction="higher_is_better",
                threshold="",
                status="ok",
                comparable_scope="comparable",
            )
        )
        rows.append(
            _scorecard_row(
                run_id=run_id,
                suite="headline_compare",
                variant=f"{VARIANT_STATEFUL}_vs_{VARIANT_BASELINE_D}",
                metric="delta_benign_off_rate",
                value=round(_safe_float(hmetrics.get("delta_benign_off_rate", 0.0)), 6),
                unit="delta_rate",
                direction="lower_is_better",
                threshold="",
                status="ok",
                comparable_scope="comparable",
            )
        )
        rows.append(
            _scorecard_row(
                run_id=run_id,
                suite="headline_compare",
                variant=f"{VARIANT_STATEFUL}_vs_{VARIANT_BASELINE_D}",
                metric="delta_steps_to_off_median",
                value=round(_safe_float(hmetrics.get("delta_steps_to_off_median", 0.0)), 6),
                unit="delta_turns",
                direction="lower_is_better",
                threshold="",
                status="ok",
                comparable_scope="comparable",
            )
        )
    return rows


def _write_scorecard(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    headers = [
        "run_id",
        "suite",
        "variant",
        "metric",
        "value",
        "unit",
        "direction",
        "threshold",
        "status",
        "comparable_scope",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in headers})


def _strict_suite_failures(
    *,
    run_eval_report: Mapping[str, Any],
    attack_layer_report: Mapping[str, Any],
) -> List[str]:
    failures: List[str] = []
    hard_fp = _safe_int((run_eval_report.get("hard_negatives", {}) or {}).get("fp", 0))
    if hard_fp != 0:
        failures.append(f"run_eval hard_negatives.fp != 0 ({hard_fp})")
    gate = attack_layer_report.get("gate", {})
    gate = dict(gate) if isinstance(gate, Mapping) else {}
    checks = gate.get("checks", [])
    if isinstance(checks, list):
        failed_checks = [str(c.get("id", "")) for c in checks if isinstance(c, Mapping) and str(c.get("status", "")).upper() != "PASS"]
        if failed_checks:
            failures.append(f"attack_layer gate has FAIL checks: {', '.join(failed_checks)}")
    return failures


def _print_summary(report: Mapping[str, Any]) -> None:
    status = str(report.get("status", "unknown"))
    headline = report.get("normalized", {}).get("headline_compare", {})
    headline = dict(headline) if isinstance(headline, Mapping) else {}
    print(f"benchmark status: {status}")
    if bool(headline.get("available")):
        metrics = headline.get("metrics", {})
        metrics = dict(metrics) if isinstance(metrics, Mapping) else {}
        print(
            "headline stateful_vs_D: "
            f"delta_attack_off_rate={_safe_float(metrics.get('delta_attack_off_rate', 0.0)):.4f}, "
            f"delta_benign_off_rate={_safe_float(metrics.get('delta_benign_off_rate', 0.0)):.4f}, "
            f"delta_steps_to_off_median={_safe_float(metrics.get('delta_steps_to_off_median', 0.0)):.2f}"
        )
    else:
        reason = str(headline.get("reason", "n/a"))
        print(f"headline stateful_vs_D unavailable: {reason}")
    artifacts = report.get("artifacts", {})
    artifacts = dict(artifacts) if isinstance(artifacts, Mapping) else {}
    print(f"report: {artifacts.get('report_json')}")
    print(f"scorecard: {artifacts.get('scorecard_csv')}")
    print(f"dataset manifest: {artifacts.get('dataset_manifest_json')}")


def _suite_failed_hard(*, suite: Mapping[str, Any], strict: bool) -> bool:
    exit_code = _safe_int(suite.get("exit_code", 1), default=1)
    has_report = isinstance(suite.get("report"), Mapping)
    if exit_code == 0 and has_report:
        return False
    if has_report and not strict:
        return False
    return True


def _suite_warning_text(suite: Mapping[str, Any]) -> Optional[str]:
    exit_code = _safe_int(suite.get("exit_code", 0), default=0)
    has_report = isinstance(suite.get("report"), Mapping)
    if exit_code != 0 and has_report:
        name = str(suite.get("name", "suite"))
        return f"{name} exited with {exit_code}, continuing with parsed report because --strict is disabled."
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Canonical benchmark orchestrator for Omega Walls.")
    parser.add_argument("--dataset-profile", default="core_oss_v1")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--mode", choices=["pi0", "hybrid", "hybrid_api"], default="pi0")
    parser.add_argument("--strict-projector", action="store_true")
    parser.add_argument("--baseline-d-enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--api-model", default="gpt-5.4-mini")
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--api-timeout-sec", type=float, default=None)
    parser.add_argument("--api-retries", type=int, default=None)
    parser.add_argument("--artifacts-root", default="artifacts/benchmark")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--allow-skip-baseline-d", action="store_true")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--dataset-registry", default="config/benchmark_datasets.yml")
    args = parser.parse_args(list(argv) if argv is not None else None)

    baseline_d_effective = bool(args.baseline_d_enable)
    if bool(args.baseline_d_enable) and not os.getenv("OPENAI_API_KEY"):
        if not bool(args.allow_skip_baseline_d):
            print(
                "baseline D requires OPENAI_API_KEY. "
                "Set OPENAI_API_KEY or pass --allow-skip-baseline-d.",
                file=sys.stderr,
            )
            return EXIT_MISSING_KEY
        baseline_d_effective = False

    snapshot = load_resolved_config(profile=str(args.profile))
    run_id = f"benchmark_{_utc_compact_now()}_{snapshot.resolved_sha256[:12]}"
    run_dir = (ROOT / str(args.artifacts_root) / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    suite_logs_dir = run_dir / "suite_logs"
    suite_logs_dir.mkdir(parents=True, exist_ok=True)

    try:
        dataset_manifest = build_dataset_manifest(
            registry_path=(ROOT / str(args.dataset_registry)).resolve(),
            dataset_profile=str(args.dataset_profile),
            seed=int(args.seed),
            runtime_mode=str(args.mode),
            config_snapshot_sha256=str(snapshot.resolved_sha256),
        )
    except Exception as exc:  # noqa: BLE001
        report = {
            "run_id": run_id,
            "generated_at_utc": _utc_now_iso(),
            "status": "failed_reproducibility",
            "error": str(exc),
        }
        report_path = run_dir / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return EXIT_FAILED

    dataset_manifest_path = run_dir / "dataset_manifest.json"
    dataset_manifest_path.write_text(json.dumps(dataset_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    support_packs_root = _prepare_support_eval_packs(run_dir=run_dir, dataset_manifest=dataset_manifest)
    suite_warnings: List[str] = []

    run_eval_argv: List[str] = [
        sys.executable,
        "scripts/run_eval.py",
        "--profile",
        str(args.profile),
        "--projector-mode",
        str(args.mode),
    ]
    if bool(args.strict_projector):
        run_eval_argv.append("--strict-projector")
    if str(args.mode) == "hybrid_api":
        run_eval_argv.extend(["--api-model", str(args.api_model)])
        if args.api_base_url:
            run_eval_argv.extend(["--api-base-url", str(args.api_base_url)])
        if args.api_timeout_sec is not None:
            run_eval_argv.extend(["--api-timeout-sec", str(args.api_timeout_sec)])
        if args.api_retries is not None:
            run_eval_argv.extend(["--api-retries", str(args.api_retries)])

    attack_layer_artifacts = run_dir / "attack_layer_suite"
    attack_argv: List[str] = [
        sys.executable,
        "scripts/run_attack_layer_cycle.py",
        "--profile",
        str(args.profile),
        "--mode",
        str(args.mode),
        "--pack-root",
        "tests/data/attack_layers/v1",
        "--seed",
        str(int(args.seed)),
        "--artifacts-root",
        str(attack_layer_artifacts),
    ]

    support_artifacts = run_dir / "support_compare_suite"
    support_argv: List[str] = [
        sys.executable,
        "scripts/eval_support_stateful_vs_stateless.py",
        "--packs-root",
        str(support_packs_root),
        "--profile",
        str(args.profile),
        "--stateful-mode",
        str(args.mode),
        "--artifacts-root",
        str(support_artifacts),
        "--seed",
        str(int(args.seed)),
    ]
    if bool(args.strict_projector):
        support_argv.append("--strict-projector")
    if str(args.mode) == "hybrid_api":
        support_argv.extend(["--api-model", str(args.api_model)])
        if args.api_base_url:
            support_argv.extend(["--api-base-url", str(args.api_base_url)])
        if args.api_timeout_sec is not None:
            support_argv.extend(["--api-timeout-sec", str(args.api_timeout_sec)])
        if args.api_retries is not None:
            support_argv.extend(["--api-retries", str(args.api_retries)])
    if baseline_d_effective:
        support_argv.append("--baseline-d-enable")
        support_argv.extend(["--baseline-d-model", str(args.api_model)])
        if args.api_base_url:
            support_argv.extend(["--baseline-d-base-url", str(args.api_base_url)])
        if args.api_timeout_sec is not None:
            support_argv.extend(["--baseline-d-timeout-sec", str(args.api_timeout_sec)])
        if args.api_retries is not None:
            support_argv.extend(["--baseline-d-retries", str(args.api_retries)])

    run_eval_suite = _run_suite_command("run_eval", run_eval_argv, suite_logs_dir)
    if _suite_failed_hard(suite=run_eval_suite, strict=bool(args.strict)):
        reason = run_eval_suite.get("parse_error") or "suite_failed"
        report = {
            "run_id": run_id,
            "generated_at_utc": _utc_now_iso(),
            "status": "failed_suite",
            "failed_suite": "run_eval",
            "reason": reason,
            "suite": run_eval_suite,
            "artifacts": {
                "dataset_manifest_json": str(dataset_manifest_path.resolve()),
                "report_json": str((run_dir / "report.json").resolve()),
            },
        }
        (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return EXIT_FAILED
    warn = _suite_warning_text(run_eval_suite)
    if warn:
        suite_warnings.append(warn)

    attack_suite = _run_suite_command("run_attack_layer_cycle", attack_argv, suite_logs_dir)
    if _suite_failed_hard(suite=attack_suite, strict=bool(args.strict)):
        reason = attack_suite.get("parse_error") or "suite_failed"
        report = {
            "run_id": run_id,
            "generated_at_utc": _utc_now_iso(),
            "status": "failed_suite",
            "failed_suite": "run_attack_layer_cycle",
            "reason": reason,
            "suite": attack_suite,
            "artifacts": {
                "dataset_manifest_json": str(dataset_manifest_path.resolve()),
                "report_json": str((run_dir / "report.json").resolve()),
            },
        }
        (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return EXIT_FAILED
    warn = _suite_warning_text(attack_suite)
    if warn:
        suite_warnings.append(warn)

    support_suite = _run_suite_command("eval_support_stateful_vs_stateless", support_argv, suite_logs_dir)
    if _suite_failed_hard(suite=support_suite, strict=bool(args.strict)):
        reason = support_suite.get("parse_error") or "suite_failed"
        report = {
            "run_id": run_id,
            "generated_at_utc": _utc_now_iso(),
            "status": "failed_suite",
            "failed_suite": "eval_support_stateful_vs_stateless",
            "reason": reason,
            "suite": support_suite,
            "artifacts": {
                "dataset_manifest_json": str(dataset_manifest_path.resolve()),
                "report_json": str((run_dir / "report.json").resolve()),
            },
        }
        (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return EXIT_FAILED
    warn = _suite_warning_text(support_suite)
    if warn:
        suite_warnings.append(warn)

    run_eval_report = dict(run_eval_suite["report"]) if isinstance(run_eval_suite["report"], Mapping) else {}
    attack_report = dict(attack_suite["report"]) if isinstance(attack_suite["report"], Mapping) else {}
    support_report = dict(support_suite["report"]) if isinstance(support_suite["report"], Mapping) else {}

    strict_failures: List[str] = []
    if bool(args.strict):
        strict_failures.extend(
            _strict_suite_failures(
                run_eval_report=run_eval_report,
                attack_layer_report=attack_report,
            )
        )
        if strict_failures:
            report = {
                "run_id": run_id,
                "generated_at_utc": _utc_now_iso(),
                "status": "failed_suite",
                "failed_suite": "strict_gate",
                "strict_failures": strict_failures,
                "suites": {
                    "run_eval": run_eval_suite,
                    "run_attack_layer_cycle": attack_suite,
                    "eval_support_stateful_vs_stateless": support_suite,
                },
                "artifacts": {
                    "dataset_manifest_json": str(dataset_manifest_path.resolve()),
                    "report_json": str((run_dir / "report.json").resolve()),
                },
            }
            (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return EXIT_FAILED

    normalized_run_eval = _normalize_run_eval(run_eval_report)
    normalized_attack = _normalize_attack_layer(attack_report)
    normalized_support = _normalize_support_compare(support_report)
    headline = _headline_compare(normalized_support)

    scorecard_rows = _build_scorecard_rows(
        run_id=run_id,
        run_eval_metrics=normalized_run_eval,
        attack_metrics=normalized_attack,
        support_metrics=normalized_support,
        headline=headline,
    )
    scorecard_path = run_dir / "scorecard.csv"
    _write_scorecard(scorecard_path, scorecard_rows)

    status = "ok"
    notes: List[str] = list(suite_warnings)
    if bool(args.baseline_d_enable) and not baseline_d_effective:
        status = "partial_ok"
        notes.append("baseline_d_skipped_due_to_missing_api_key")
    if bool(args.baseline_d_enable) and baseline_d_effective and not bool(headline.get("available")):
        status = "partial_ok"
        notes.append("headline_stateful_vs_d_unavailable")

    report = {
        "run_id": run_id,
        "generated_at_utc": _utc_now_iso(),
        "status": status,
        "notes": notes,
        "config": {
            "dataset_profile": str(args.dataset_profile),
            "profile": str(args.profile),
            "mode": str(args.mode),
            "strict_projector": bool(args.strict_projector),
            "strict": bool(args.strict),
            "seed": int(args.seed),
            "baseline_d_requested": bool(args.baseline_d_enable),
            "baseline_d_effective": bool(baseline_d_effective),
            "allow_skip_baseline_d": bool(args.allow_skip_baseline_d),
            "api_model": str(args.api_model),
            "api_base_url": (str(args.api_base_url) if args.api_base_url else None),
            "api_timeout_sec": (float(args.api_timeout_sec) if args.api_timeout_sec is not None else None),
            "api_retries": (int(args.api_retries) if args.api_retries is not None else None),
            "config_snapshot_sha256": str(snapshot.resolved_sha256),
        },
        "reproducibility": {
            "strict_manifest": True,
            "dataset_manifest_path": str(dataset_manifest_path.resolve()),
            "dataset_count": int(dataset_manifest.get("dataset_count", 0)),
        },
        "datasets": dataset_manifest,
        "suites": {
            "run_eval": run_eval_suite,
            "run_attack_layer_cycle": attack_suite,
            "eval_support_stateful_vs_stateless": support_suite,
        },
        "normalized": {
            "run_eval": normalized_run_eval,
            "attack_layer": normalized_attack,
            "support_compare": normalized_support,
            "headline_compare": headline,
        },
        "artifacts": {
            "report_json": str((run_dir / "report.json").resolve()),
            "scorecard_csv": str(scorecard_path.resolve()),
            "dataset_manifest_json": str(dataset_manifest_path.resolve()),
            "suite_logs_dir": str(suite_logs_dir.resolve()),
        },
    }

    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _print_summary(report)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
