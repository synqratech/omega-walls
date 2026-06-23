from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.api import server as api_server
from omega.config.loader import load_resolved_config
from omega.env_file import load_repo_env_file


DEFAULT_MODES = ("vision_single",)
DEFAULT_CONCURRENCY_GRID = (1,)


@dataclass(frozen=True)
class ImageSample:
    path: Path
    sha256: str
    dropped_duplicates: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImageEvalRow:
    mode: str
    repeat: int
    phase: str
    worker_id: int
    worker_ordinal: int
    is_worker_first_request: bool
    name: str
    sha256: str
    verdict: str
    risk_score: int
    latency_ms: float
    semantic_status: str
    vision_status: str
    ocr_status: str
    ocr_provider: str
    ocr_quality_status: str
    ocr_kept_spans: int
    ocr_dropped_low_confidence: int
    ocr_geometry_coverage_ratio: float
    ocr_active_walls: List[str]
    ocr_adjudication_result: str
    region_trigger_reason: str
    region_variant_count: int
    provider_call_count: int
    retry_count: int
    cache_hit_last_request: bool
    vision_fallback_used: bool
    second_pass_attempted: bool
    second_pass_result: str
    first_pass_latency_ms: float | None
    second_pass_latency_ms: float | None
    semantic_latency_ms: float | None
    token_usage: Dict[str, Any]
    reasons: List[str]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _resolve_key(file_path: str | None) -> None:
    load_repo_env_file()
    if os.environ.get("OPENAI_API_KEY"):
        return
    if not file_path:
        return
    raw = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"(sk-[A-Za-z0-9_\-]+)", raw)
    key = str(match.group(1)) if match else ""
    if key:
        os.environ["OPENAI_API_KEY"] = key


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _opaque_run_namespace(name: str) -> str:
    return hashlib.sha1(str(name).encode("utf-8")).hexdigest()[:10]


def _deep_merge_dict(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dict(dict(current), dict(value))
        else:
            merged[key] = value
    return merged


def _iter_image_files(root: Path, include_stems: List[str] | None = None) -> List[Path]:
    def _sort_key(path: Path) -> tuple[int, str]:
        stem = str(path.stem)
        if stem.isdigit():
            return (int(stem), stem)
        return (10**9, stem)

    patterns: Iterable[str] = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
    files: List[Path] = []
    for pattern in patterns:
        files.extend(root.glob(pattern))
    files = sorted({path.resolve(): path for path in files}.values(), key=_sort_key)
    if include_stems:
        allow = {str(x).strip() for x in include_stems if str(x).strip()}
        files = [path for path in files if str(path.stem) in allow]
    return files


def collect_image_samples(
    root: Path,
    *,
    max_samples: int,
    include_stems: List[str] | None = None,
    dedupe_by_sha256: bool = True,
) -> tuple[List[ImageSample], List[Dict[str, Any]]]:
    files = _iter_image_files(root, include_stems=include_stems)
    if not dedupe_by_sha256:
        samples = [ImageSample(path=path, sha256=_file_sha256(path)) for path in files[: max(1, int(max_samples))]]
        return samples, []

    by_sha: Dict[str, Dict[str, Any]] = {}
    ordered_shas: List[str] = []
    target = max(1, int(max_samples))
    for path in files:
        sha = _file_sha256(path)
        if sha not in by_sha:
            by_sha[sha] = {"kept": path, "dropped": []}
            ordered_shas.append(sha)
        else:
            by_sha[sha]["dropped"].append(path.name)
        if len(ordered_shas) >= target:
            continue

    selected_shas = ordered_shas[:target]
    samples = [
        ImageSample(
            path=Path(by_sha[sha]["kept"]),
            sha256=str(sha),
            dropped_duplicates=tuple(str(x) for x in list(by_sha[sha]["dropped"])),
        )
        for sha in selected_shas
    ]
    duplicates = [
        {
            "sha256": str(sha),
            "kept": str(by_sha[sha]["kept"].name),
            "dropped": [str(x) for x in list(by_sha[sha]["dropped"])],
        }
        for sha in selected_shas
        if by_sha[sha]["dropped"]
    ]
    return samples, duplicates


def _mode_overrides(mode: str, *, ocr_provider: str) -> Dict[str, Any]:
    mode_name = str(mode).strip().lower()
    image_region = False
    ocr_enabled = "false"
    if mode_name == "vision_region":
        image_region = True
    elif mode_name == "vision_plus_ocr":
        ocr_enabled = "true"
    elif mode_name == "vision_region_plus_ocr":
        image_region = True
        ocr_enabled = "true"
    elif mode_name != "vision_single":
        raise ValueError(f"unsupported mode: {mode}")
    return {
        "projector": {
            "mode": "hybrid_api",
            "api_perception": {
                "enabled": "true",
                "image_region_pass_enabled": bool(image_region),
                "image_region_pass": {
                    "enabled": bool(image_region),
                    "trigger_mode": "uncertain",
                    "pressure_abs_max": 0.35,
                    "confidence_max": 0.72,
                    "max_tiles": 5,
                    "overlap_ratio": 0.10,
                    "include_center_crop": True,
                },
                "provider_options": {
                    "capabilities": {
                        "text": True,
                        "image": True,
                    }
                },
            },
        },
        "retriever": {
            "sqlite_fts": {
                "attachments": {
                    "ocr": {
                        "enabled": str(ocr_enabled),
                        "provider": str(ocr_provider),
                    }
                }
            }
        },
    }


def _build_runtime(
    *,
    profile: str,
    mode: str,
    ocr_provider: str,
    pi0_semantic_enabled: str,
    pi0_semantic_device: str,
    cache_namespace: str,
    api_runtime_mode: str,
    attachment_visual_enabled: str,
) -> Any:
    overrides: Dict[str, Any] = {
        "runtime": {"guard_mode": "enforce"},
        "pi0": {
            "semantic": {
                "enabled": str(pi0_semantic_enabled),
                "device": str(pi0_semantic_device),
            }
        },
    }
    if str(api_runtime_mode).strip():
        overrides["api"] = {"runtime": {"mode": str(api_runtime_mode).strip()}}
    if str(attachment_visual_enabled).strip():
        overrides["retriever"] = {
            "sqlite_fts": {
                "attachments": {
                    "visual": {
                        "enabled": str(attachment_visual_enabled).strip(),
                    }
                }
            }
        }
    mode_cfg = _mode_overrides(mode, ocr_provider=ocr_provider)
    cache_slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(cache_namespace)).strip("_") or "default"
    projector_cfg = mode_cfg.setdefault("projector", {})
    api_cfg = projector_cfg.setdefault("api_perception", {})
    api_cfg["cache_path"] = f"artifacts/live_image_eval_release/cache/{cache_slug}/{mode}/cache.jsonl"
    api_cfg["error_log_path"] = f"artifacts/live_image_eval_release/cache/{cache_slug}/{mode}/errors.jsonl"
    overrides = _deep_merge_dict(overrides, mode_cfg)
    snapshot = load_resolved_config(profile=profile, cli_overrides=overrides)
    return api_server._make_runtime(snapshot.resolved)


def _projector_semantic_status(runtime: Any) -> Dict[str, Any]:
    projector = getattr(runtime, "projector", None)
    status_fn = getattr(projector, "semantic_status", None)
    if callable(status_fn):
        try:
            return dict(status_fn() or {})
        except Exception as exc:  # noqa: BLE001
            return {"status_error": str(exc)}
    return {}


def _runtime_probe() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        import torch

        out["torch"] = {
            "version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "cuda_version": str(torch.version.cuda),
            "device_name": (str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else ""),
        }
    except Exception as exc:  # noqa: BLE001
        out["torch"] = {"error": str(exc)}
    try:
        import onnxruntime as ort

        out["onnxruntime"] = {
            "version": str(ort.__version__),
            "providers": [str(x) for x in list(ort.get_available_providers() or [])],
        }
    except Exception as exc:  # noqa: BLE001
        out["onnxruntime"] = {"error": str(exc)}
    return out


def _rapidocr_probe() -> Dict[str, Any]:
    try:
        from rapidocr import RapidOCR  # type: ignore

        t0 = time.perf_counter()
        engine = RapidOCR()
        t1 = time.perf_counter()
        return {
            "status": "ok",
            "init_ms": round((t1 - t0) * 1000.0, 2),
            "engine_type": str(type(engine).__name__),
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}


def _scan_image(
    runtime: Any,
    sample: ImageSample,
    *,
    idx: int,
    run_namespace: str,
    mode_label: str,
    repeat: int,
    phase: str,
    worker_id: int,
    worker_ordinal: int,
) -> ImageEvalRow:
    suffix = str(sample.path.suffix).strip().lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(suffix, "application/octet-stream")
    payload = {
        "tenant_id": f"wainject-image-eval-{run_namespace}-{idx:03d}",
        "request_id": f"{run_namespace}-{mode_label}-{phase}-r{repeat:02d}-w{worker_id:02d}-{idx:03d}-{sample.path.stem}",
        "session_id": f"img-eval-{run_namespace}-r{repeat:02d}-w{worker_id:02d}-s{idx:03d}",
        "use_extracted_text": False,
        "file_bytes": sample.path.read_bytes(),
        "filename": sample.path.name,
        "mime": mime,
    }
    started_at = time.perf_counter()
    out = api_server._scan_request(runtime, payload)
    latency_ms = (time.perf_counter() - started_at) * 1000.0
    trace = out.get("policy_trace", {}) if isinstance(out.get("policy_trace", {}), dict) else {}
    ocr_quality = trace.get("ocr_quality", {}) if isinstance(trace.get("ocr_quality", {}), dict) else {}
    return ImageEvalRow(
        mode=str(mode_label),
        repeat=int(repeat),
        phase=str(phase),
        worker_id=int(worker_id),
        worker_ordinal=int(worker_ordinal),
        is_worker_first_request=bool(worker_ordinal == 1),
        name=sample.path.name,
        sha256=str(sample.sha256),
        verdict=str(out.get("verdict", "")),
        risk_score=int(out.get("risk_score", 0)),
        latency_ms=round(latency_ms, 2),
        semantic_status=str(trace.get("semantic_failure_status", "")),
        vision_status=str(trace.get("vision_semantic_status", "")),
        ocr_status=str(trace.get("ocr_status", "none")),
        ocr_provider=str(trace.get("ocr_provider", "")),
        ocr_quality_status=str(ocr_quality.get("status", "none")),
        ocr_kept_spans=int(ocr_quality.get("kept_spans", 0) or 0),
        ocr_dropped_low_confidence=int(ocr_quality.get("dropped_low_confidence", 0) or 0),
        ocr_geometry_coverage_ratio=float(ocr_quality.get("geometry_coverage_ratio", 0.0) or 0.0),
        ocr_active_walls=[str(x) for x in list(trace.get("ocr_modality_active_walls", []) or [])],
        ocr_adjudication_result=str(trace.get("ocr_adjudication_result", "not_attempted")),
        region_trigger_reason=str(trace.get("region_trigger_reason", "not_evaluated")),
        region_variant_count=int(trace.get("region_variant_count", 0) or 0),
        provider_call_count=int(trace.get("provider_call_count", 0) or 0),
        retry_count=int(trace.get("retry_count", 0) or 0),
        cache_hit_last_request=bool(trace.get("cache_hit_last_request", False)),
        vision_fallback_used=bool(trace.get("vision_fallback_used", False)),
        second_pass_attempted=bool(trace.get("second_pass_attempted", False)),
        second_pass_result=str(trace.get("second_pass_result", "not_attempted")),
        first_pass_latency_ms=(
            float(trace["first_pass_latency_ms"]) if trace.get("first_pass_latency_ms") is not None else None
        ),
        second_pass_latency_ms=(
            float(trace["second_pass_latency_ms"]) if trace.get("second_pass_latency_ms") is not None else None
        ),
        semantic_latency_ms=(
            float(trace["semantic_latency_ms"]) if trace.get("semantic_latency_ms") is not None else None
        ),
        token_usage=dict(trace.get("token_usage", {}) or {}),
        reasons=[str(x) for x in list(out.get("reasons", []) or [])],
    )


def _percentile(values: Sequence[float], pct: float) -> float:
    rows = sorted(float(x) for x in values)
    if not rows:
        return 0.0
    if len(rows) == 1:
        return float(rows[0])
    rank = (len(rows) - 1) * float(pct)
    lo = int(rank)
    hi = min(lo + 1, len(rows) - 1)
    frac = rank - lo
    return float(rows[lo] + (rows[hi] - rows[lo]) * frac)


def _sum_token_usage(rows: Sequence[ImageEvalRow]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for row in rows:
        usage = row.token_usage if isinstance(row.token_usage, dict) else {}
        for key, value in usage.items():
            name = str(key)
            if isinstance(value, dict):
                current = merged.get(name, {})
                current_map = dict(current) if isinstance(current, dict) else {}
                for nested_key, nested_value in value.items():
                    nested_name = str(nested_key)
                    if isinstance(nested_value, (int, float)):
                        current_map[nested_name] = float(current_map.get(nested_name, 0.0)) + float(nested_value)
                    else:
                        current_map[nested_name] = nested_value
                merged[name] = current_map
            elif isinstance(value, (int, float)):
                merged[name] = float(merged.get(name, 0.0)) + float(value)
            else:
                merged[name] = value
    return merged


def summarize_rows(rows: Sequence[ImageEvalRow]) -> Dict[str, Any]:
    rows_list = list(rows)
    total = len(rows_list)
    detected = [row for row in rows_list if row.verdict in {"block", "quarantine"}]
    allows = [row for row in rows_list if row.verdict == "allow"]
    latency_values = [float(row.latency_ms) for row in rows_list]
    semantic_latency_values = [float(row.semantic_latency_ms) for row in rows_list if row.semantic_latency_ms is not None]
    provider_call_values = [int(row.provider_call_count) for row in rows_list]
    retry_values = [int(row.retry_count) for row in rows_list]
    ocr_kept_values = [int(row.ocr_kept_spans) for row in rows_list]
    return {
        "total": total,
        "detected_rate": (len(detected) / float(total)) if total else 0.0,
        "allow_rate": (len(allows) / float(total)) if total else 0.0,
        "avg_risk": (sum(row.risk_score for row in rows_list) / float(total)) if total else 0.0,
        "verdict_counts": {
            "allow": sum(1 for row in rows_list if row.verdict == "allow"),
            "quarantine": sum(1 for row in rows_list if row.verdict == "quarantine"),
            "block": sum(1 for row in rows_list if row.verdict == "block"),
        },
        "semantic_failed_count": sum(1 for row in rows_list if row.semantic_status == "semantic_failed"),
        "vision_active_count": sum(1 for row in rows_list if row.vision_status == "vision_semantic_active"),
        "semantic_latency_active_count": sum(1 for row in rows_list if row.semantic_latency_ms is not None),
        "ocr_success_count": sum(1 for row in rows_list if row.ocr_status == "success"),
        "ocr_quality_ok_count": sum(1 for row in rows_list if row.ocr_quality_status in {"ok", "filtered"}),
        "ocr_kept_spans": {
            "total": sum(ocr_kept_values),
            "avg": (sum(ocr_kept_values) / float(total)) if total else 0.0,
            "max": max(ocr_kept_values) if ocr_kept_values else 0,
        },
        "ocr_dropped_low_confidence_total": sum(row.ocr_dropped_low_confidence for row in rows_list),
        "ocr_adjudication_counts": {
            status: sum(1 for row in rows_list if row.ocr_adjudication_result == status)
            for status in sorted({row.ocr_adjudication_result for row in rows_list})
        },
        "region_trigger_counts": {
            reason: sum(1 for row in rows_list if row.region_trigger_reason == reason)
            for reason in sorted({row.region_trigger_reason for row in rows_list})
        },
        "region_variant_count_total": sum(row.region_variant_count for row in rows_list),
        "vision_fallback_used_count": sum(1 for row in rows_list if row.vision_fallback_used),
        "second_pass_attempted_count": sum(1 for row in rows_list if row.second_pass_attempted),
        "cache_hit_last_request_count": sum(1 for row in rows_list if row.cache_hit_last_request),
        "latency_ms": {
            "avg": (sum(latency_values) / float(total)) if total else 0.0,
            "p50": _percentile(latency_values, 0.50),
            "p95": _percentile(latency_values, 0.95),
            "p99": _percentile(latency_values, 0.99),
        },
        "semantic_latency_ms": {
            "avg": (sum(semantic_latency_values) / float(len(semantic_latency_values))) if semantic_latency_values else 0.0,
            "p50": _percentile(semantic_latency_values, 0.50),
            "p95": _percentile(semantic_latency_values, 0.95),
            "p99": _percentile(semantic_latency_values, 0.99),
        },
        "provider_call_count": {
            "avg": (sum(provider_call_values) / float(total)) if total else 0.0,
            "max": max(provider_call_values) if provider_call_values else 0,
            "active_count": sum(1 for row in rows_list if int(row.provider_call_count) > 0),
            "total": sum(provider_call_values),
        },
        "retry_count": {
            "avg": (sum(retry_values) / float(total)) if total else 0.0,
            "max": max(retry_values) if retry_values else 0,
        },
        "token_usage": _sum_token_usage(rows_list),
    }


def summarize_phase_metrics(rows: Sequence[ImageEvalRow]) -> Dict[str, Any]:
    rows_list = list(rows)
    fresh_rows = [row for row in rows_list if row.phase == "fresh"]
    cache_rows = [row for row in rows_list if row.phase == "cache_replay"]
    cold_rows = [row for row in fresh_rows if row.is_worker_first_request]
    warm_fresh_rows = [
        row
        for row in fresh_rows
        if (not row.is_worker_first_request) and int(row.provider_call_count) > 0
    ]
    cache_hit_rows = [row for row in cache_rows if row.cache_hit_last_request]
    return {
        "quality_fresh": summarize_rows(fresh_rows),
        "cold_first_request": summarize_rows(cold_rows),
        "warm_fresh_provider_calls": summarize_rows(warm_fresh_rows),
        "cache_hit_replay": summarize_rows(cache_hit_rows),
    }


def _run_runtime_pass(
    runtime: Any,
    samples: Sequence[ImageSample],
    *,
    mode: str,
    repeat: int,
    phase: str,
    run_namespace: str,
    worker_id: int,
) -> List[ImageEvalRow]:
    out: List[ImageEvalRow] = []
    for idx, sample in enumerate(samples, start=1):
        row = _scan_image(
            runtime,
            sample,
            idx=idx,
            run_namespace=run_namespace,
            mode_label=mode,
            repeat=repeat,
            phase=phase,
            worker_id=worker_id,
            worker_ordinal=idx,
        )
        out.append(row)
    return out


def _run_worker_scenario(
    *,
    profile: str,
    mode: str,
    ocr_provider: str,
    pi0_semantic_enabled: str,
    pi0_semantic_device: str,
    api_runtime_mode: str,
    attachment_visual_enabled: str,
    samples: Sequence[ImageSample],
    repeat: int,
    cache_namespace: str,
    worker_id: int,
    measure_cache_replay: bool,
) -> Dict[str, Any]:
    runtime = _build_runtime(
        profile=profile,
        mode=mode,
        ocr_provider=ocr_provider,
        pi0_semantic_enabled=pi0_semantic_enabled,
        pi0_semantic_device=pi0_semantic_device,
        cache_namespace=f"{cache_namespace}_w{worker_id:02d}",
        api_runtime_mode=api_runtime_mode,
        attachment_visual_enabled=attachment_visual_enabled,
    )
    run_namespace = _opaque_run_namespace(f"{cache_namespace}_w{worker_id:02d}")
    status_before = _projector_semantic_status(runtime)
    fresh_rows = _run_runtime_pass(
        runtime,
        samples,
        mode=mode,
        repeat=repeat,
        phase="fresh",
        run_namespace=run_namespace,
        worker_id=worker_id,
    )
    status_after_fresh = _projector_semantic_status(runtime)
    cache_rows: List[ImageEvalRow] = []
    if measure_cache_replay:
        cache_rows = _run_runtime_pass(
            runtime,
            samples,
            mode=mode,
            repeat=repeat,
            phase="cache_replay",
            run_namespace=run_namespace,
            worker_id=worker_id,
        )
    status_after_cache = _projector_semantic_status(runtime)
    return {
        "status_before": status_before,
        "status_after_fresh": status_after_fresh,
        "status_after_cache": status_after_cache,
        "rows": fresh_rows + cache_rows,
    }


def _partition_samples(samples: Sequence[ImageSample], concurrency: int) -> List[List[ImageSample]]:
    groups: List[List[ImageSample]] = [[] for _ in range(max(1, int(concurrency)))]
    for idx, sample in enumerate(samples):
        groups[idx % len(groups)].append(sample)
    return [group for group in groups if group]


def _run_scenario(
    *,
    profile: str,
    mode: str,
    samples: Sequence[ImageSample],
    repeat: int,
    concurrency: int,
    ocr_provider: str,
    pi0_semantic_enabled: str,
    pi0_semantic_device: str,
    api_runtime_mode: str,
    attachment_visual_enabled: str,
    cache_namespace: str,
    measure_cache_replay: bool,
) -> Dict[str, Any]:
    batches = _partition_samples(samples, max(1, int(concurrency)))
    if int(concurrency) <= 1:
        return _run_worker_scenario(
            profile=profile,
            mode=mode,
            ocr_provider=ocr_provider,
            pi0_semantic_enabled=pi0_semantic_enabled,
            pi0_semantic_device=pi0_semantic_device,
            api_runtime_mode=api_runtime_mode,
            attachment_visual_enabled=attachment_visual_enabled,
            samples=list(samples),
            repeat=repeat,
            cache_namespace=cache_namespace,
            worker_id=0,
            measure_cache_replay=measure_cache_replay,
        )

    rows: List[ImageEvalRow] = []
    status_before_list: List[Dict[str, Any]] = []
    status_after_fresh_list: List[Dict[str, Any]] = []
    status_after_cache_list: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(batches)) as executor:
        futures = [
            executor.submit(
                _run_worker_scenario,
                profile=profile,
                mode=mode,
                ocr_provider=ocr_provider,
                pi0_semantic_enabled=pi0_semantic_enabled,
                pi0_semantic_device=pi0_semantic_device,
                api_runtime_mode=api_runtime_mode,
                attachment_visual_enabled=attachment_visual_enabled,
                samples=batch,
                repeat=repeat,
                cache_namespace=cache_namespace,
                worker_id=worker_id,
                measure_cache_replay=measure_cache_replay,
            )
            for worker_id, batch in enumerate(batches)
        ]
        for future in concurrent.futures.as_completed(futures):
            payload = future.result()
            rows.extend(list(payload["rows"]))
            status_before_list.append(dict(payload["status_before"]))
            status_after_fresh_list.append(dict(payload["status_after_fresh"]))
            status_after_cache_list.append(dict(payload["status_after_cache"]))
    rows.sort(key=lambda row: (row.phase, row.worker_id, row.worker_ordinal, row.name))
    return {
        "status_before": {"workers": status_before_list},
        "status_after_fresh": {"workers": status_after_fresh_list},
        "status_after_cache": {"workers": status_after_cache_list},
        "rows": rows,
    }


def _write_report(
    *,
    report_path: Path,
    profile: str,
    root: Path,
    requested_max_samples: int,
    unique_samples: Sequence[ImageSample],
    duplicates: Sequence[Dict[str, Any]],
    include_stems: List[str],
    modes: List[str],
    repeats: int,
    concurrency_grid: List[int],
    ocr_provider: str,
    pi0_semantic_enabled: str,
    pi0_semantic_device: str,
    api_runtime_mode: str,
    attachment_visual_enabled: str,
    cache_namespace_base: str,
    runtime_probe: Dict[str, Any],
    rapidocr_probe: Dict[str, Any],
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    report = {
        "benchmark": f"image_slice::{root.parent.name}/{root.name}",
        "profile": str(profile),
        "root": str(root),
        "requested_max_samples": int(requested_max_samples),
        "unique_sample_count": int(len(list(unique_samples))),
        "unique_samples": [
            {
                "name": sample.path.name,
                "sha256": sample.sha256,
                "dropped_duplicates": list(sample.dropped_duplicates),
            }
            for sample in list(unique_samples)
        ],
        "duplicates": list(duplicates),
        "include_stems": include_stems,
        "modes": list(modes),
        "repeats": int(repeats),
        "concurrency_grid": [int(x) for x in list(concurrency_grid)],
        "ocr_provider": str(ocr_provider),
        "pi0_semantic_enabled": str(pi0_semantic_enabled),
        "pi0_semantic_device": str(pi0_semantic_device),
        "api_runtime_mode": str(api_runtime_mode),
        "attachment_visual_enabled": str(attachment_visual_enabled),
        "cache_namespace_base": str(cache_namespace_base),
        "runtime_probe": dict(runtime_probe),
        "rapidocr_probe": dict(rapidocr_probe),
        "scenarios": list(results),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate image semantic modes on a WAInject image slice.")
    parser.add_argument("--profile", default="pilot")
    parser.add_argument("--root", default="data/WAInjectBench/image/malicious/EIA")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--include-stems", default="")
    parser.add_argument("--api-key-file", default="API_OpenAI.txt")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency-grid", default=",".join(str(x) for x in DEFAULT_CONCURRENCY_GRID))
    parser.add_argument("--ocr-provider", default="rapidocr")
    parser.add_argument("--pi0-semantic-enabled", default="true")
    parser.add_argument("--pi0-semantic-device", default="auto")
    parser.add_argument("--api-runtime-mode", default="")
    parser.add_argument("--attachment-visual-enabled", default="")
    parser.add_argument("--cache-namespace", default="")
    parser.add_argument("--artifacts-root", default="artifacts/live_image_pilot_gate")
    parser.add_argument("--dedupe-sha256", default="true")
    parser.add_argument("--measure-cache-replay", default="true")
    args = parser.parse_args()

    _resolve_key(str(args.api_key_file) if args.api_key_file else None)
    root = (ROOT / str(args.root)).resolve()
    include_stems = [part.strip() for part in str(args.include_stems or "").split(",") if part.strip()]
    modes = [part.strip() for part in str(args.modes or "").split(",") if part.strip()]
    if not modes:
        raise SystemExit("at least one mode is required")
    concurrency_grid = [max(1, int(part.strip())) for part in str(args.concurrency_grid or "1").split(",") if part.strip()]
    if not concurrency_grid:
        raise SystemExit("at least one concurrency value is required")
    dedupe_sha256 = str(args.dedupe_sha256).strip().lower() not in {"0", "false", "no", "off"}
    measure_cache_replay = str(args.measure_cache_replay).strip().lower() not in {"0", "false", "no", "off"}
    unique_samples, duplicates = collect_image_samples(
        root,
        max_samples=int(args.max_samples),
        include_stems=include_stems or None,
        dedupe_by_sha256=dedupe_sha256,
    )
    if not unique_samples:
        raise SystemExit(f"no supported image files found under {root}")

    out_dir = (ROOT / str(args.artifacts_root) / f"image_ocr_slice_{_utc_stamp()}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"

    runtime_probe = _runtime_probe()
    rapidocr_probe = _rapidocr_probe()
    results: List[Dict[str, Any]] = []
    cache_namespace_base = str(args.cache_namespace).strip() or f"{root.parent.name}_{root.name}"

    for mode in modes:
        for repeat in range(1, max(1, int(args.repeats)) + 1):
            for concurrency in concurrency_grid:
                cache_namespace = f"{cache_namespace_base}_{mode}_r{repeat:02d}_c{concurrency:02d}"
                scenario = _run_scenario(
                    profile=str(args.profile),
                    mode=mode,
                    samples=unique_samples,
                    repeat=repeat,
                    concurrency=int(concurrency),
                    ocr_provider=str(args.ocr_provider),
                    pi0_semantic_enabled=str(args.pi0_semantic_enabled),
                    pi0_semantic_device=str(args.pi0_semantic_device),
                    api_runtime_mode=str(args.api_runtime_mode),
                    attachment_visual_enabled=str(args.attachment_visual_enabled),
                    cache_namespace=cache_namespace,
                    measure_cache_replay=measure_cache_replay,
                )
                rows = list(scenario["rows"])
                scenario_report = {
                    "mode": str(mode),
                    "repeat": int(repeat),
                    "concurrency": int(concurrency),
                    "semantic_status_before": dict(scenario["status_before"]),
                    "semantic_status_after_fresh": dict(scenario["status_after_fresh"]),
                    "semantic_status_after_cache_replay": dict(scenario["status_after_cache"]),
                    "phase_metrics": summarize_phase_metrics(rows),
                    "rows": [asdict(row) for row in rows],
                }
                results.append(scenario_report)
                for row in rows:
                    print(
                        json.dumps(
                            {
                                "mode": mode,
                                "repeat": repeat,
                                "concurrency": concurrency,
                                "phase": row.phase,
                                "name": row.name,
                                "verdict": row.verdict,
                                "risk_score": row.risk_score,
                                "latency_ms": row.latency_ms,
                                "provider_calls": row.provider_call_count,
                                "cache_hit": row.cache_hit_last_request,
                                "second_pass": row.second_pass_result,
                            },
                            ensure_ascii=False,
                        )
                    )

    report = _write_report(
        report_path=report_path,
        profile=str(args.profile),
        root=root,
        requested_max_samples=int(args.max_samples),
        unique_samples=unique_samples,
        duplicates=duplicates,
        include_stems=include_stems,
        modes=modes,
        repeats=max(1, int(args.repeats)),
        concurrency_grid=concurrency_grid,
        ocr_provider=str(args.ocr_provider),
        pi0_semantic_enabled=str(args.pi0_semantic_enabled),
        pi0_semantic_device=str(args.pi0_semantic_device),
        api_runtime_mode=str(args.api_runtime_mode),
        attachment_visual_enabled=str(args.attachment_visual_enabled),
        cache_namespace_base=cache_namespace_base,
        runtime_probe=runtime_probe,
        rapidocr_probe=rapidocr_probe,
        results=results,
    )
    print(
        json.dumps(
            {
                "report_json": str(report_path),
                "out_dir": str(out_dir),
                "modes": list(report["modes"]),
                "unique_sample_count": int(report["unique_sample_count"]),
                "scenario_count": int(len(list(report["scenarios"]))),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
