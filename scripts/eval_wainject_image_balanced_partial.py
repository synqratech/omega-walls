from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
BENIGN_CATEGORY_TARGETS = {
    "embedded_img": 25,
    "screenshot": 25,
}
MALICIOUS_CATEGORIES = (
    "EIA",
    "popup",
    "VPI",
    "VWA_adv_embedded_img",
    "VWA_adv_screenshot",
    "wasp",
    "WebInject",
)


@dataclass(frozen=True)
class SelectedImage:
    label: str
    category: str
    source_path: Path
    staged_name: str


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _iter_image_files(root: Path) -> List[Path]:
    files = [path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files, key=lambda path: path.name.lower())


def _shuffle_select(paths: Sequence[Path], *, sample_count: int, seed: int) -> List[Path]:
    items = list(paths)
    rng = random.Random(int(seed))
    rng.shuffle(items)
    return sorted(items[: max(0, int(sample_count))], key=lambda path: path.name.lower())


def _opaque_staged_name(*, index: int, suffix: str) -> str:
    return f"sample_{int(index):06d}{suffix.lower()}"


def _cli_path_arg(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved)


def _allocate_proportional_targets(counts: Dict[str, int], *, total: int) -> Dict[str, int]:
    requested_total = max(0, int(total))
    positive = {key: max(0, int(value)) for key, value in counts.items()}
    available_total = sum(positive.values())
    target_total = min(requested_total, available_total)
    if target_total <= 0 or available_total <= 0:
        return {key: 0 for key in counts}

    raw = {key: (target_total * positive[key] / float(available_total)) for key in counts}
    allocated = {key: min(positive[key], int(math.floor(raw[key]))) for key in counts}
    remaining = target_total - sum(allocated.values())

    while remaining > 0:
        candidates = [
            (
                raw[key] - allocated[key],
                positive[key] - allocated[key],
                key,
            )
            for key in counts
            if allocated[key] < positive[key]
        ]
        if not candidates:
            break
        _, _, best_key = max(candidates, key=lambda item: (item[0], item[1], item[2]))
        allocated[best_key] += 1
        remaining -= 1

    return allocated


def _select_benign_samples(root: Path, *, seed: int) -> List[SelectedImage]:
    selected: List[SelectedImage] = []
    global_index = 0
    for offset, (category, target) in enumerate(BENIGN_CATEGORY_TARGETS.items(), start=1):
        category_root = root / category
        files = _iter_image_files(category_root)
        if len(files) < target:
            raise ValueError(f"benign category {category} has only {len(files)} files, expected at least {target}")
        chosen = _shuffle_select(files, sample_count=target, seed=int(seed) + offset)
        for path in chosen:
            global_index += 1
            selected.append(
                SelectedImage(
                    label="benign",
                    category=category,
                    source_path=path,
                    staged_name=_opaque_staged_name(index=global_index, suffix=path.suffix),
                )
            )
    return selected


def _select_malicious_samples(root: Path, *, seed: int, total: int) -> List[SelectedImage]:
    counts = {category: len(_iter_image_files(root / category)) for category in MALICIOUS_CATEGORIES}
    allocation = _allocate_proportional_targets(counts, total=total)
    selected: List[SelectedImage] = []
    global_index = 0
    for offset, category in enumerate(MALICIOUS_CATEGORIES, start=101):
        files = _iter_image_files(root / category)
        quota = allocation.get(category, 0)
        chosen = _shuffle_select(files, sample_count=quota, seed=int(seed) + offset)
        for path in chosen:
            global_index += 1
            selected.append(
                SelectedImage(
                    label="malicious",
                    category=category,
                    source_path=path,
                    staged_name=_opaque_staged_name(index=global_index, suffix=path.suffix),
                )
            )
    return selected


def _stage_samples(stage_root: Path, samples: Sequence[SelectedImage]) -> None:
    stage_root.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        shutil.copy2(sample.source_path, stage_root / sample.staged_name)


def _build_backend_command(
    *,
    profile: str,
    root: Path,
    max_samples: int,
    artifacts_root: Path,
    mode: str,
    ocr_provider: str,
    pi0_semantic_enabled: str,
    pi0_semantic_device: str,
    api_runtime_mode: str,
    attachment_visual_enabled: str,
    cache_namespace: str,
) -> List[str]:
    command = [
        str(ROOT / ".venv" / "Scripts" / "python.exe"),
        str(ROOT / "scripts" / "eval_wainject_image_ocr_slice.py"),
        "--profile",
        str(profile),
        "--root",
        _cli_path_arg(root),
        "--max-samples",
        str(int(max_samples)),
        "--modes",
        str(mode),
        "--repeats",
        "1",
        "--concurrency-grid",
        "1",
        "--ocr-provider",
        str(ocr_provider),
        "--pi0-semantic-enabled",
        str(pi0_semantic_enabled),
        "--pi0-semantic-device",
        str(pi0_semantic_device),
        "--dedupe-sha256",
        "false",
        "--measure-cache-replay",
        "false",
        "--artifacts-root",
        _cli_path_arg(artifacts_root),
    ]
    if str(api_runtime_mode).strip():
        command.extend(["--api-runtime-mode", str(api_runtime_mode).strip()])
    if str(attachment_visual_enabled).strip():
        command.extend(["--attachment-visual-enabled", str(attachment_visual_enabled).strip()])
    if str(cache_namespace).strip():
        command.extend(["--cache-namespace", str(cache_namespace).strip()])
    return command


def _find_latest_report(run_root: Path) -> Path:
    candidates = sorted(run_root.glob("image_ocr_slice_*/report.json"))
    if not candidates:
        raise FileNotFoundError(f"no backend reports found under {run_root}")
    return candidates[-1]


def _run_backend(
    *,
    profile: str,
    stage_root: Path,
    max_samples: int,
    artifacts_root: Path,
    mode: str,
    ocr_provider: str,
    pi0_semantic_enabled: str,
    pi0_semantic_device: str,
    api_runtime_mode: str,
    attachment_visual_enabled: str,
    cache_namespace: str,
) -> Dict[str, Any]:
    command = _build_backend_command(
        profile=profile,
        root=stage_root,
        max_samples=max_samples,
        artifacts_root=artifacts_root,
        mode=mode,
        ocr_provider=ocr_provider,
        pi0_semantic_enabled=pi0_semantic_enabled,
        pi0_semantic_device=pi0_semantic_device,
        api_runtime_mode=api_runtime_mode,
        attachment_visual_enabled=attachment_visual_enabled,
        cache_namespace=cache_namespace,
    )
    proc = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    report_path = _find_latest_report(artifacts_root)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "report_path": str(report_path),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "report": payload,
    }


def _validate_selected_total(samples: Sequence[SelectedImage], *, expected_total: int, label: str) -> None:
    actual_total = len(list(samples))
    if actual_total != int(expected_total):
        raise ValueError(f"{label} sample selection produced {actual_total}, expected exactly {expected_total}")


def _extract_quality_fresh(report: Dict[str, Any]) -> Dict[str, Any]:
    scenarios = list(report.get("scenarios") or [])
    if not scenarios:
        raise ValueError("backend report has no scenarios")
    return dict((scenarios[0].get("phase_metrics") or {}).get("quality_fresh") or {})


def _first_scenario(report: Dict[str, Any]) -> Dict[str, Any]:
    scenarios = list(report.get("scenarios") or [])
    if not scenarios:
        raise ValueError("backend report has no scenarios")
    return dict(scenarios[0] or {})


def _normalize_semantic_status_block(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        workers = value.get("workers")
        if isinstance(workers, list) and workers:
            first = workers[0]
            if isinstance(first, dict):
                return dict(first)
        return dict(value)
    return {}


def _validate_backend_quality_total(
    report: Dict[str, Any],
    *,
    expected_total: int,
    label: str,
) -> Dict[str, Any]:
    quality = _extract_quality_fresh(report)
    actual_total = int(quality.get("total", 0) or 0)
    if actual_total != int(expected_total):
        raise ValueError(
            f"{label} backend quality_fresh.total={actual_total} does not match expected selected total {expected_total}"
        )
    return quality


def _summarize_label_quality(quality: Dict[str, Any], *, label: str) -> Dict[str, Any]:
    verdict_counts = dict(quality.get("verdict_counts") or {})
    total = max(0, int(quality.get("total", 0) or 0))
    detected = int(verdict_counts.get("block", 0) or 0) + int(verdict_counts.get("quarantine", 0) or 0)
    allow = int(verdict_counts.get("allow", 0) or 0)
    return {
        "label": label,
        "sample_count": total,
        "detected_count": detected,
        "allow_count": allow,
        "detect_rate": (detected / float(total)) if total else 0.0,
        "allow_rate": (allow / float(total)) if total else 0.0,
        "verdict_counts": verdict_counts,
        "ocr_success_count": int(quality.get("ocr_success_count", 0) or 0),
        "vision_active_count": int(quality.get("vision_active_count", 0) or 0),
        "semantic_failed_count": int(quality.get("semantic_failed_count", 0) or 0),
        "semantic_latency_active_count": int(quality.get("semantic_latency_active_count", 0) or 0),
        "provider_call_active_count": int(((quality.get("provider_call_count") or {}).get("active_count", 0)) or 0),
        "provider_call_total": int(((quality.get("provider_call_count") or {}).get("total", 0)) or 0),
        "cache_hit_count": int(quality.get("cache_hit_last_request_count", 0) or 0),
        "avg_latency_ms": float(((quality.get("latency_ms") or {}).get("avg", 0.0)) or 0.0),
    }


def _build_breakdown(samples: Sequence[SelectedImage]) -> Dict[str, Any]:
    by_category: Dict[str, Dict[str, Any]] = {}
    for sample in samples:
        bucket = by_category.setdefault(
            sample.category,
            {
                "sample_count": 0,
                "staged_names": [],
                "source_files": [],
            },
        )
        bucket["sample_count"] += 1
        bucket["staged_names"].append(sample.staged_name)
        bucket["source_files"].append(str(sample.source_path))
    return by_category


def _combine_operational(benign: Dict[str, Any], malicious: Dict[str, Any]) -> Dict[str, Any]:
    total = int(benign.get("sample_count", 0)) + int(malicious.get("sample_count", 0))
    latency_weighted = (
        float(benign.get("avg_latency_ms", 0.0)) * int(benign.get("sample_count", 0))
        + float(malicious.get("avg_latency_ms", 0.0)) * int(malicious.get("sample_count", 0))
    )
    return {
        "ocr_success_count": int(benign.get("ocr_success_count", 0)) + int(malicious.get("ocr_success_count", 0)),
        "vision_active_count": int(benign.get("vision_active_count", 0)) + int(malicious.get("vision_active_count", 0)),
        "semantic_failed_count": int(benign.get("semantic_failed_count", 0))
        + int(malicious.get("semantic_failed_count", 0)),
        "semantic_latency_active_count": int(benign.get("semantic_latency_active_count", 0))
        + int(malicious.get("semantic_latency_active_count", 0)),
        "provider_call_active_count": int(benign.get("provider_call_active_count", 0))
        + int(malicious.get("provider_call_active_count", 0)),
        "provider_call_total": int(benign.get("provider_call_total", 0))
        + int(malicious.get("provider_call_total", 0)),
        "cache_hit_count": int(benign.get("cache_hit_count", 0))
        + int(malicious.get("cache_hit_count", 0)),
        "avg_latency_ms": (latency_weighted / float(total)) if total else 0.0,
        "total": total,
    }


def _extract_run_audit(report: Dict[str, Any]) -> Dict[str, Any]:
    scenario = _first_scenario(report)
    after_fresh = _normalize_semantic_status_block(scenario.get("semantic_status_after_fresh"))
    quality = _extract_quality_fresh(report)
    provider_counts = dict(quality.get("provider_call_count") or {})
    return {
        "api_semantic_mode": str(after_fresh.get("api_semantic_mode", "") or ""),
        "api_provider": str(after_fresh.get("api_provider", "") or ""),
        "api_model": str(after_fresh.get("api_model", "") or ""),
        "provider_id": str(after_fresh.get("provider_id", "") or ""),
        "fallback_level": str(after_fresh.get("fallback_level", "") or ""),
        "fallback_reason": str(after_fresh.get("fallback_reason", "") or ""),
        "api_adapter_active": bool(after_fresh.get("api_adapter_active", False)),
        "pi0_enabled_mode": str(after_fresh.get("enabled_mode", "") or ""),
        "pi0_active": bool(after_fresh.get("active", False)),
        "pi0_attempted": bool(after_fresh.get("attempted", False)),
        "vision_active_count": int(quality.get("vision_active_count", 0) or 0),
        "ocr_success_count": int(quality.get("ocr_success_count", 0) or 0),
        "provider_call_active_count": int(provider_counts.get("active_count", 0) or 0),
        "provider_call_total": int(provider_counts.get("total", 0) or 0),
        "cache_hit_count": int(quality.get("cache_hit_last_request_count", 0) or 0),
        "semantic_latency_active_count": int(quality.get("semantic_latency_active_count", 0) or 0),
    }


def _build_quality_gates(
    *,
    mode: str,
    requested_pi0_semantic_enabled: str,
    requested_api_runtime_mode: str,
    cache_replay_expected: bool,
    benign_audit: Dict[str, Any],
    malicious_audit: Dict[str, Any],
    benign_total: int,
    malicious_total: int,
) -> Dict[str, Any]:
    checks = []
    if str(mode) == "vision_single":
        checks.extend(
            [
                {
                    "name": "api_semantic_mode_hybrid_cloud",
                    "passed": str(benign_audit.get("api_semantic_mode")) == "hybrid_cloud"
                    and str(malicious_audit.get("api_semantic_mode")) == "hybrid_cloud",
                    "expected": "hybrid_cloud",
                    "actual": {
                        "benign": benign_audit.get("api_semantic_mode"),
                        "malicious": malicious_audit.get("api_semantic_mode"),
                    },
                },
                {
                    "name": "vision_active_full_coverage",
                    "passed": int(benign_audit.get("vision_active_count", 0)) == int(benign_total)
                    and int(malicious_audit.get("vision_active_count", 0)) == int(malicious_total),
                    "expected": {
                        "benign": int(benign_total),
                        "malicious": int(malicious_total),
                    },
                    "actual": {
                        "benign": benign_audit.get("vision_active_count"),
                        "malicious": malicious_audit.get("vision_active_count"),
                    },
                },
                {
                    "name": "semantic_latency_full_coverage",
                    "passed": int(benign_audit.get("semantic_latency_active_count", 0)) == int(benign_total)
                    and int(malicious_audit.get("semantic_latency_active_count", 0)) == int(malicious_total),
                    "expected": {
                        "benign": int(benign_total),
                        "malicious": int(malicious_total),
                    },
                    "actual": {
                        "benign": benign_audit.get("semantic_latency_active_count"),
                        "malicious": malicious_audit.get("semantic_latency_active_count"),
                    },
                },
                {
                    "name": "provider_or_cache_full_coverage",
                    "passed": (
                        int(benign_audit.get("provider_call_active_count", 0))
                        + int(benign_audit.get("cache_hit_count", 0))
                        == int(benign_total)
                    )
                    and (
                        int(malicious_audit.get("provider_call_active_count", 0))
                        + int(malicious_audit.get("cache_hit_count", 0))
                        == int(malicious_total)
                    ),
                    "expected": {
                        "benign": int(benign_total),
                        "malicious": int(malicious_total),
                    },
                    "actual": {
                        "benign": int(benign_audit.get("provider_call_active_count", 0))
                        + int(benign_audit.get("cache_hit_count", 0)),
                        "malicious": int(malicious_audit.get("provider_call_active_count", 0))
                        + int(malicious_audit.get("cache_hit_count", 0)),
                    },
                },
                {
                    "name": "cache_hits_match_mode",
                    "passed": (
                        (
                            bool(cache_replay_expected)
                            and int(benign_audit.get("cache_hit_count", 0)) >= 0
                            and int(malicious_audit.get("cache_hit_count", 0)) >= 0
                        )
                        or (
                            not bool(cache_replay_expected)
                            and int(benign_audit.get("cache_hit_count", 0)) == 0
                            and int(malicious_audit.get("cache_hit_count", 0)) == 0
                        )
                    ),
                    "expected": "cache_hits_zero_when_live_uncached",
                    "actual": {
                        "benign": benign_audit.get("cache_hit_count"),
                        "malicious": malicious_audit.get("cache_hit_count"),
                        "cache_replay_expected": bool(cache_replay_expected),
                    },
                },
                {
                    "name": "ocr_off_for_headline",
                    "passed": int(benign_audit.get("ocr_success_count", 0)) == 0
                    and int(malicious_audit.get("ocr_success_count", 0)) == 0,
                    "expected": 0,
                    "actual": {
                        "benign": benign_audit.get("ocr_success_count"),
                        "malicious": malicious_audit.get("ocr_success_count"),
                    },
                },
                {
                    "name": "pi0_semantic_inactive",
                    "passed": str(requested_pi0_semantic_enabled).strip().lower() in {"false", "0", "no", "off"}
                    and not bool(benign_audit.get("pi0_active", False))
                    and not bool(malicious_audit.get("pi0_active", False)),
                    "expected": "requested_off_and_inactive",
                    "actual": {
                        "requested": str(requested_pi0_semantic_enabled),
                        "benign_active": benign_audit.get("pi0_active"),
                        "malicious_active": malicious_audit.get("pi0_active"),
                    },
                },
                {
                    "name": "runtime_stateless",
                    "passed": str(requested_api_runtime_mode).strip().lower() == "stateless",
                    "expected": "stateless",
                    "actual": str(requested_api_runtime_mode),
                },
            ]
        )
    return {
        "passed": all(bool(check.get("passed", False)) for check in checks) if checks else True,
        "checks": checks,
    }


def build_balanced_report(
    *,
    profile: str,
    seed: int,
    mode: str,
    ocr_provider: str,
    pi0_semantic_enabled: str,
    pi0_semantic_device: str,
    api_runtime_mode: str,
    attachment_visual_enabled: str,
    cache_replay_expected: bool,
    benign_samples: Sequence[SelectedImage],
    malicious_samples: Sequence[SelectedImage],
    benign_run: Dict[str, Any],
    malicious_run: Dict[str, Any],
) -> Dict[str, Any]:
    benign_expected = len(list(benign_samples))
    malicious_expected = len(list(malicious_samples))
    benign_quality = _validate_backend_quality_total(
        dict(benign_run["report"]),
        expected_total=benign_expected,
        label="benign",
    )
    malicious_quality = _validate_backend_quality_total(
        dict(malicious_run["report"]),
        expected_total=malicious_expected,
        label="malicious",
    )
    benign_summary = _summarize_label_quality(benign_quality, label="benign")
    malicious_summary = _summarize_label_quality(malicious_quality, label="malicious")
    operational = _combine_operational(benign_summary, malicious_summary)
    benign_audit = _extract_run_audit(dict(benign_run["report"]))
    malicious_audit = _extract_run_audit(dict(malicious_run["report"]))
    quality_gates = _build_quality_gates(
        mode=str(mode),
        requested_pi0_semantic_enabled=str(pi0_semantic_enabled),
        requested_api_runtime_mode=str(api_runtime_mode),
        cache_replay_expected=bool(cache_replay_expected),
        benign_audit=benign_audit,
        malicious_audit=malicious_audit,
        benign_total=benign_expected,
        malicious_total=malicious_expected,
    )
    return {
        "benchmark": "WAInjectBench:image:balanced_partial",
        "profile": str(profile),
        "mode": str(mode),
        "ocr_provider": str(ocr_provider),
        "pi0_semantic_enabled": str(pi0_semantic_enabled),
        "pi0_semantic_device": str(pi0_semantic_device),
        "api_runtime_mode": str(api_runtime_mode),
        "attachment_visual_enabled": str(attachment_visual_enabled),
        "seed": int(seed),
        "sampling": {
            "strategy": "balanced_label_category_partial",
            "sample_count": int(len(list(benign_samples)) + len(list(malicious_samples))),
            "benign": {
                "target_total": 50,
                "selected_total": benign_expected,
                "targets_by_category": dict(BENIGN_CATEGORY_TARGETS),
            },
            "malicious": {
                "target_total": 50,
                "selected_total": malicious_expected,
                "categories": list(MALICIOUS_CATEGORIES),
            },
        },
        "summary": {
            "sample_count": int(benign_expected + malicious_expected),
            "attack_detect_rate": float(malicious_summary["detect_rate"]),
            "benign_false_positive_rate": float(benign_summary["detect_rate"]),
            "malicious_allow_rate": float(malicious_summary["allow_rate"]),
            "benign_allow_rate": float(benign_summary["allow_rate"]),
            "operational_health": operational,
        },
        "audit": {
            "benign": benign_audit,
            "malicious": malicious_audit,
            "quality_gates": quality_gates,
        },
        "benign_breakdown": {
            "sampling": _build_breakdown(benign_samples),
            "quality_fresh": benign_summary,
        },
        "malicious_breakdown": {
            "sampling": _build_breakdown(malicious_samples),
            "quality_fresh": malicious_summary,
        },
        "raw_runs": {
            "benign": {
                "report_path": str(benign_run["report_path"]),
            },
            "malicious": {
                "report_path": str(malicious_run["report_path"]),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a balanced partial WAInjectBench image eval via the existing OCR slice backend.")
    parser.add_argument("--profile", default="prod_vision")
    parser.add_argument("--root", default="data/WAInjectBench/image")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--benign-total", type=int, default=50)
    parser.add_argument("--malicious-total", type=int, default=50)
    parser.add_argument("--mode", default="vision_plus_ocr")
    parser.add_argument("--ocr-provider", default="rapidocr")
    parser.add_argument("--pi0-semantic-enabled", default="true")
    parser.add_argument("--pi0-semantic-device", default="auto")
    parser.add_argument("--api-runtime-mode", default="")
    parser.add_argument("--attachment-visual-enabled", default="")
    parser.add_argument("--measure-cache-replay", default="false")
    parser.add_argument("--artifacts-root", default="artifacts/wainject_partial_image")
    args = parser.parse_args()

    if int(args.benign_total) != 50:
        raise SystemExit("benign-total must stay at 50 for the balanced partial contract")
    if int(args.malicious_total) != 50:
        raise SystemExit("malicious-total must stay at 50 for the balanced partial contract")

    dataset_root = (ROOT / str(args.root)).resolve()
    benign_root = dataset_root / "benign"
    malicious_root = dataset_root / "malicious"
    out_dir = (ROOT / str(args.artifacts_root) / f"wainject_image_balanced_partial_{_utc_stamp()}").resolve()
    stage_root = out_dir / "staging"
    backend_root = out_dir / "backend_runs"
    run_slug = out_dir.name
    benign_stage = stage_root / "benign"
    malicious_stage = stage_root / "malicious"
    benign_backend_root = backend_root / "benign"
    malicious_backend_root = backend_root / "malicious"

    benign_samples = _select_benign_samples(benign_root, seed=int(args.seed))
    malicious_samples = _select_malicious_samples(malicious_root, seed=int(args.seed), total=int(args.malicious_total))
    _validate_selected_total(benign_samples, expected_total=int(args.benign_total), label="benign")
    _validate_selected_total(malicious_samples, expected_total=int(args.malicious_total), label="malicious")
    _stage_samples(benign_stage, benign_samples)
    _stage_samples(malicious_stage, malicious_samples)

    benign_run = _run_backend(
        profile=str(args.profile),
        stage_root=benign_stage,
        max_samples=int(args.benign_total),
        artifacts_root=benign_backend_root,
        mode=str(args.mode),
        ocr_provider=str(args.ocr_provider),
        pi0_semantic_enabled=str(args.pi0_semantic_enabled),
        pi0_semantic_device=str(args.pi0_semantic_device),
        api_runtime_mode=str(args.api_runtime_mode),
        attachment_visual_enabled=str(args.attachment_visual_enabled),
        cache_namespace=f"{run_slug}_benign",
    )
    malicious_run = _run_backend(
        profile=str(args.profile),
        stage_root=malicious_stage,
        max_samples=int(args.malicious_total),
        artifacts_root=malicious_backend_root,
        mode=str(args.mode),
        ocr_provider=str(args.ocr_provider),
        pi0_semantic_enabled=str(args.pi0_semantic_enabled),
        pi0_semantic_device=str(args.pi0_semantic_device),
        api_runtime_mode=str(args.api_runtime_mode),
        attachment_visual_enabled=str(args.attachment_visual_enabled),
        cache_namespace=f"{run_slug}_malicious",
    )

    report = build_balanced_report(
        profile=str(args.profile),
        seed=int(args.seed),
        mode=str(args.mode),
        ocr_provider=str(args.ocr_provider),
        pi0_semantic_enabled=str(args.pi0_semantic_enabled),
        pi0_semantic_device=str(args.pi0_semantic_device),
        api_runtime_mode=str(args.api_runtime_mode),
        attachment_visual_enabled=str(args.attachment_visual_enabled),
        cache_replay_expected=str(args.measure_cache_replay).strip().lower() not in {"0", "false", "no", "off"},
        benign_samples=benign_samples,
        malicious_samples=malicious_samples,
        benign_run=benign_run,
        malicious_run=malicious_run,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not bool(((report.get("audit") or {}).get("quality_gates") or {}).get("passed", False)):
        print(json.dumps({"status": "invalid_quality_gate", "report_path": str(report_path)}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": "ok", "report_path": str(report_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
