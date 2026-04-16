"""Dataset builder for PiTheta fine-tuning."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from urllib import error as urlerror
from urllib import request as urlrequest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import yaml

from omega.config.loader import load_resolved_config
from omega.eval.deepset_adapter import build_deepset_samples
from omega.interfaces.contracts_v1 import ContentItem, WALLS_V1
from omega.projector.pi0_intent_v2 import Pi0IntentAwareV2
from redteam.generator import generate
from redteam.runner import load_jsonl

PI_THETA_SCHEMA_VERSION = "pitheta_dataset_v2"
WALL_ORDER = list(WALLS_V1)
SOURCE_TRUST_VALUES = {"trusted", "semi_trusted", "untrusted"}
CHUNK_BUCKET_VALUES = {"64", "128_256", "512"}
CONTENT_FILTER_MODE_OFF = "off"
CONTENT_FILTER_MODE_HEURISTIC = "heuristic"
CONTENT_FILTER_MODE_OPENAI = "openai"
CONTENT_FILTER_MODE_OPENAI_THEN_HEURISTIC = "openai_then_heuristic"
CONTENT_FILTER_MODES = {
    CONTENT_FILTER_MODE_OFF,
    CONTENT_FILTER_MODE_HEURISTIC,
    CONTENT_FILTER_MODE_OPENAI,
    CONTENT_FILTER_MODE_OPENAI_THEN_HEURISTIC,
}
OPENAI_MODERATION_BLOCK_CATEGORIES_DEFAULT = (
    "violence",
    "violence/graphic",
    "harassment/threatening",
    "hate/threatening",
    "self-harm/instructions",
    "self-harm/intent",
    "sexual/minors",
    "illicit/violent",
)
HEURISTIC_HARD_BLOCK_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:how to|instructions?|guide|steps?)\b.{0,64}\b(?:kill|murder|assassinat(?:e|ion)|shoot|stab)\b.{0,32}\b(?:person|people|someone|him|her|them|crowd|civilian)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:make|build|assemble|craft)\b.{0,32}\b(?:bomb|explosive|ied|pipe bomb|molotov)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:how to|instructions?|guide|steps?)\b.{0,64}\b(?:make|build|assemble)\b.{0,32}\b(?:bomb|explosive|ied|pipe bomb|molotov|silencer|ghost gun)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:mass shooting|school shooting|terror attack|suicide attack|car bomb)\b",
        flags=re.IGNORECASE,
    ),
)
HEURISTIC_HARD_BLOCK_ALLOW_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\bkill\s+process(?:es)?\b", flags=re.IGNORECASE),
    re.compile(r"\bterminate\s+process(?:es)?\b", flags=re.IGNORECASE),
)


class APIRequestError(RuntimeError):
    def __init__(self, *, code: int, body: str):
        self.code = int(code)
        self.body = str(body)
        super().__init__(f"HTTP {self.code}: {self.body}")


@dataclass(frozen=True)
class ContentFilterDecision:
    blocked: bool
    provider: str
    reason: str
    categories: Tuple[str, ...]
    error: str = ""


@dataclass
class DatasetContentFilter:
    mode: str = CONTENT_FILTER_MODE_OFF
    fail_closed: bool = False
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str = "https://api.openai.com/v1"
    model: str = "omni-moderation-latest"
    timeout_sec: float = 20.0
    max_retries: int = 2
    backoff_sec: float = 0.75
    block_categories: Tuple[str, ...] = OPENAI_MODERATION_BLOCK_CATEGORIES_DEFAULT
    block_score_threshold: float = 0.0
    apply_splits: Tuple[str, ...] = ("train", "dev", "holdout")
    log_path: Optional[Path] = None
    _cache: Dict[str, ContentFilterDecision] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mode_norm = str(self.mode or CONTENT_FILTER_MODE_OFF).strip().lower()
        if mode_norm not in CONTENT_FILTER_MODES:
            raise ValueError(f"invalid content filter mode: {self.mode}")
        self.mode = mode_norm
        self.base_url = str(self.base_url or "https://api.openai.com/v1").rstrip("/")
        self.timeout_sec = max(1.0, float(self.timeout_sec))
        self.max_retries = max(0, int(self.max_retries))
        self.backoff_sec = max(0.0, float(self.backoff_sec))
        self.block_score_threshold = max(0.0, float(self.block_score_threshold))
        self.block_categories = tuple(str(x).strip() for x in list(self.block_categories or ()) if str(x).strip())
        apply = []
        for split in list(self.apply_splits or ()):
            try:
                apply.append(_sanitize_split(str(split)))
            except ValueError:
                continue
        self.apply_splits = tuple(sorted(set(apply))) if apply else ("train", "dev", "holdout")

    @property
    def active(self) -> bool:
        return self.mode != CONTENT_FILTER_MODE_OFF

    def should_apply_to_split(self, split: str) -> bool:
        return _sanitize_split(split) in set(self.apply_splits)

    def _cache_key(self, text: str) -> str:
        return _sha256_bytes(str(text).encode("utf-8"))

    def _heuristic_decision(self, text: str) -> ContentFilterDecision:
        norm = " ".join(str(text or "").strip().split())
        if not norm:
            return ContentFilterDecision(blocked=False, provider="heuristic", reason="", categories=tuple())
        for allow_pattern in HEURISTIC_HARD_BLOCK_ALLOW_PATTERNS:
            if allow_pattern.search(norm):
                return ContentFilterDecision(blocked=False, provider="heuristic", reason="", categories=tuple())
        for idx, pattern in enumerate(HEURISTIC_HARD_BLOCK_PATTERNS, start=1):
            if pattern.search(norm):
                return ContentFilterDecision(
                    blocked=True,
                    provider="heuristic",
                    reason=f"violent_or_weapon_pattern_{idx}",
                    categories=("violent_or_weapon",),
                )
        return ContentFilterDecision(blocked=False, provider="heuristic", reason="", categories=tuple())

    def _api_key(self) -> str:
        return str(os.getenv(self.api_key_env, "")).strip()

    def _moderate_openai(self, text: str) -> ContentFilterDecision:
        api_key = self._api_key()
        if not api_key:
            raise RuntimeError(f"missing_api_key_env:{self.api_key_env}")
        url = self.base_url + "/moderations"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "input": str(text)}

        def _is_retryable_http(code: int) -> bool:
            c = int(code)
            return c in {408, 409, 429} or c >= 500

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = _post_json(url=url, payload=payload, headers=headers, timeout_sec=self.timeout_sec)
                results = resp.get("results", [])
                first = results[0] if isinstance(results, list) and results else {}
                if not isinstance(first, Mapping):
                    raise ValueError("moderation response missing results[0]")
                flagged = bool(first.get("flagged", False))
                categories_obj = first.get("categories", {})
                scores_obj = first.get("category_scores", {})
                categories_map = categories_obj if isinstance(categories_obj, Mapping) else {}
                scores_map = scores_obj if isinstance(scores_obj, Mapping) else {}
                matched: List[str] = []
                for name in self.block_categories:
                    cat_flag = categories_map.get(name)
                    if isinstance(cat_flag, bool) and cat_flag:
                        matched.append(name)
                        continue
                    if self.block_score_threshold > 0.0:
                        try:
                            score = float(scores_map.get(name, 0.0))
                        except Exception:  # noqa: BLE001
                            score = 0.0
                        if score >= self.block_score_threshold:
                            matched.append(f"{name}@{score:.3f}")
                if flagged and not matched:
                    matched.append("flagged")
                matched_sorted = tuple(sorted(set(matched)))
                return ContentFilterDecision(
                    blocked=bool(matched_sorted),
                    provider="openai_moderation",
                    reason=",".join(matched_sorted) if matched_sorted else "",
                    categories=matched_sorted,
                )
            except APIRequestError as exc:
                last_exc = exc
                if (not _is_retryable_http(exc.code)) or attempt >= self.max_retries:
                    break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= self.max_retries:
                    break
            if attempt < self.max_retries:
                time.sleep(self.backoff_sec * (2**attempt))
        raise RuntimeError(f"openai_moderation_failed:{last_exc}")

    def decide(self, text: str) -> ContentFilterDecision:
        cache_key = self._cache_key(text)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        if self.mode == CONTENT_FILTER_MODE_HEURISTIC:
            decision = self._heuristic_decision(text)
            self._cache[cache_key] = decision
            return decision
        if self.mode == CONTENT_FILTER_MODE_OPENAI:
            try:
                decision = self._moderate_openai(text)
            except Exception as exc:  # noqa: BLE001
                if self.fail_closed:
                    decision = ContentFilterDecision(
                        blocked=True,
                        provider="openai_moderation",
                        reason="moderation_api_error_fail_closed",
                        categories=("moderation_api_error",),
                        error=str(exc),
                    )
                else:
                    decision = ContentFilterDecision(
                        blocked=False,
                        provider="openai_moderation",
                        reason="",
                        categories=tuple(),
                        error=str(exc),
                    )
            self._cache[cache_key] = decision
            return decision

        # openai_then_heuristic
        try:
            decision = self._moderate_openai(text)
        except Exception as exc:  # noqa: BLE001
            if self.fail_closed:
                decision = ContentFilterDecision(
                    blocked=True,
                    provider="openai_moderation",
                    reason="moderation_api_error_fail_closed",
                    categories=("moderation_api_error",),
                    error=str(exc),
                )
            else:
                fallback = self._heuristic_decision(text)
                if fallback.blocked:
                    decision = fallback
                else:
                    decision = ContentFilterDecision(
                        blocked=False,
                        provider="openai_then_heuristic",
                        reason="",
                        categories=tuple(),
                        error=str(exc),
                    )
        self._cache[cache_key] = decision
        return decision


def _post_json(*, url: str, payload: Mapping[str, Any], headers: Mapping[str, str], timeout_sec: float) -> Dict[str, Any]:
    data = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
    req = urlrequest.Request(url=url, data=data, headers=dict(headers), method="POST")
    try:
        with urlrequest.urlopen(req, timeout=float(timeout_sec)) as resp:
            raw = resp.read().decode("utf-8")
    except urlerror.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            body = str(exc)
        raise APIRequestError(code=int(exc.code), body=body) from exc
    except urlerror.URLError as exc:
        raise RuntimeError(f"url_error: {exc}") from exc
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("api response is not a JSON object")
    return parsed


def _append_jsonl_row(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(row), ensure_ascii=True) + "\n")


@dataclass(frozen=True)
class PiThetaRecord:
    sample_id: str
    text: str
    wall_labels: List[int]
    pressure_level: List[int]
    polarity: List[int]
    source: str
    source_type: str
    source_trust: str
    lang: str
    split: str
    label_quality: str
    is_attack: int
    chunk_bucket: str
    approx_tokens: int


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _jsonl_write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(dict(row), ensure_ascii=True) for row in rows)
    path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")


def _jsonl_read(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            out.append(row)
    return out


def _load_registry(path: str) -> Dict[str, Any]:
    registry = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(registry, dict):
        raise ValueError("pitheta dataset registry must be a mapping")
    if "pitheta_dataset_registry" in registry:
        nested = registry.get("pitheta_dataset_registry")
        if not isinstance(nested, dict):
            raise ValueError("pitheta_dataset_registry must be a mapping")
        registry = nested
    datasets = registry.get("datasets", [])
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("pitheta dataset registry must define non-empty datasets")
    return registry


def _sanitize_split(split: str) -> str:
    split_norm = str(split).strip().lower()
    if split_norm not in {"train", "dev", "holdout"}:
        raise ValueError("split must be train|dev|holdout")
    return split_norm


def _estimate_token_count(text: str) -> int:
    tokens = [tok for tok in str(text).strip().split() if tok]
    return max(1, len(tokens))


def _chunk_bucket_from_tokens(token_count: int) -> str:
    n = max(1, int(token_count))
    if n <= 96:
        return "64"
    if n <= 320:
        return "128_256"
    return "512"


def _normalize_record(record: Mapping[str, Any], *, split_override: str | None = None) -> Dict[str, Any]:
    out = dict(record)
    text_norm = str(out.get("text", "")).strip()
    out["text"] = text_norm
    out["source"] = str(out.get("source", "unknown"))
    out["source_type"] = str(out.get("source_type", "other"))
    out["source_trust"] = str(out.get("source_trust", "untrusted"))
    if split_override is not None:
        out["split"] = _sanitize_split(split_override)
    else:
        out["split"] = _sanitize_split(str(out.get("split", "train")))

    wall_labels = [int(x) for x in list(out.get("wall_labels", [0, 0, 0, 0]))]
    pressure_level = [int(x) for x in list(out.get("pressure_level", [0, 0, 0, 0]))]
    polarity = [int(x) for x in list(out.get("polarity", [0, 0, 0, 0]))]
    while len(wall_labels) < 4:
        wall_labels.append(0)
    while len(pressure_level) < 4:
        pressure_level.append(0)
    while len(polarity) < 4:
        polarity.append(0)
    wall_labels = wall_labels[:4]
    pressure_level = pressure_level[:4]
    polarity = polarity[:4]
    for idx in range(4):
        # Canonical label rule: no wall activation => neutral polarity and zero pressure.
        if int(wall_labels[idx]) == 0:
            pressure_level[idx] = 0
            polarity[idx] = 0
    out["wall_labels"] = wall_labels
    out["pressure_level"] = pressure_level
    out["polarity"] = polarity

    approx_tokens = int(out.get("approx_tokens") or _estimate_token_count(text_norm))
    out["approx_tokens"] = approx_tokens
    out["chunk_bucket"] = str(out.get("chunk_bucket") or _chunk_bucket_from_tokens(approx_tokens))
    return out


def _validate_record(record: Mapping[str, Any]) -> None:
    required = {
        "sample_id",
        "text",
        "wall_labels",
        "pressure_level",
        "polarity",
        "source",
        "lang",
        "split",
        "label_quality",
        "is_attack",
    }
    missing = sorted(required.difference(record.keys()))
    if missing:
        raise ValueError(f"PiTheta record is missing required keys: {missing}")
    if not str(record["sample_id"]).strip():
        raise ValueError("sample_id must be non-empty")
    if not str(record["text"]).strip():
        raise ValueError(f"text must be non-empty for sample_id={record['sample_id']}")
    wall_labels = list(record["wall_labels"])
    pressure_level = list(record["pressure_level"])
    polarity = list(record["polarity"])
    if len(wall_labels) != 4 or len(pressure_level) != 4 or len(polarity) != 4:
        raise ValueError("wall_labels/pressure_level/polarity must have length 4")
    for value in wall_labels:
        if int(value) not in {0, 1}:
            raise ValueError("wall_labels values must be 0|1")
    for value in pressure_level:
        if int(value) not in {0, 1, 2, 3}:
            raise ValueError("pressure_level values must be 0|1|2|3")
    for value in polarity:
        if int(value) not in {-1, 0, 1}:
            raise ValueError("polarity values must be -1|0|1")
    _sanitize_split(str(record["split"]))
    if int(record["is_attack"]) not in {0, 1}:
        raise ValueError("is_attack must be 0|1")
    source_type = str(record.get("source_type", "")).strip()
    if not source_type:
        raise ValueError("source_type must be non-empty")
    source_trust = str(record.get("source_trust", "")).strip()
    if source_trust not in SOURCE_TRUST_VALUES:
        raise ValueError(f"source_trust must be one of {sorted(SOURCE_TRUST_VALUES)}")
    chunk_bucket = str(record.get("chunk_bucket", "")).strip()
    if chunk_bucket not in CHUNK_BUCKET_VALUES:
        raise ValueError(f"chunk_bucket must be one of {sorted(CHUNK_BUCKET_VALUES)}")
    approx_tokens = int(record.get("approx_tokens", 0))
    if approx_tokens <= 0:
        raise ValueError("approx_tokens must be > 0")
    for idx in range(4):
        if int(wall_labels[idx]) == 0 and (int(pressure_level[idx]) != 0 or int(polarity[idx]) != 0):
            raise ValueError("when wall_labels[k]==0 both pressure_level[k] and polarity[k] must be 0")


def _stable_record_hash(record: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(record), ensure_ascii=True, sort_keys=True).encode("utf-8")
    return _sha256_bytes(payload)


def _iter_jsonl_rows(path: Path) -> Iterable[Dict[str, Any]]:
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            yield row


def _quantize_pressure(raw: float, bins: Sequence[float]) -> int:
    b0, b1, b2 = [float(x) for x in bins]
    if raw <= b0:
        return 1
    if raw <= b1:
        return 2
    if raw <= b2:
        return 3
    return 3


def _weak_label_attack(
    projector: Pi0IntentAwareV2,
    sample_id: str,
    source: str,
    text: str,
    *,
    ordinal_bins: Sequence[float],
) -> Tuple[List[int], List[int], List[int]]:
    projection = projector.project(
        ContentItem(
            doc_id=sample_id,
            source_id=source,
            source_type="other",
            trust="untrusted",
            text=text,
        )
    )
    wall_labels = [1 if float(v) > 0.0 else 0 for v in projection.v.tolist()]
    polarity = [int(x) for x in projection.evidence.polarity]
    raw_scores = [float(x) for x in projection.evidence.debug_scores_raw]
    pressure_level = [
        _quantize_pressure(raw_scores[i], ordinal_bins) if int(wall_labels[i]) == 1 else 0 for i in range(4)
    ]
    if sum(wall_labels) == 0:
        wall_labels = [1, 0, 0, 0]
        polarity = [1, 0, 0, 0]
        pressure_level = [2, 0, 0, 0]
    return wall_labels, pressure_level, polarity


def _build_record(
    *,
    sample_id: str,
    text: str,
    source: str,
    source_type: str,
    source_trust: str,
    split: str,
    is_attack: int,
    projector: Pi0IntentAwareV2,
    label_quality: str,
    ordinal_bins: Sequence[float],
    active_floor_gold: int,
    pressure_level_override: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    split_norm = _sanitize_split(split)
    text_norm = str(text).strip()
    if not text_norm:
        raise ValueError(f"empty text for sample_id={sample_id}")

    if int(is_attack) == 1:
        if pressure_level_override is not None:
            wall_labels, _, polarity = _weak_label_attack(
                projector,
                sample_id=sample_id,
                source=source,
                text=text_norm,
                ordinal_bins=ordinal_bins,
            )
            pressure_level = [int(x) for x in list(pressure_level_override)]
        else:
            wall_labels, pressure_level, polarity = _weak_label_attack(
                projector,
                sample_id=sample_id,
                source=source,
                text=text_norm,
                ordinal_bins=ordinal_bins,
            )
    else:
        wall_labels = [0, 0, 0, 0]
        pressure_level = [0, 0, 0, 0]
        polarity = [0, 0, 0, 0]

    if str(label_quality).lower() == "gold":
        for i in range(4):
            if int(wall_labels[i]) == 1:
                pressure_level[i] = max(int(pressure_level[i]), int(active_floor_gold))

    record = {
        "sample_id": str(sample_id),
        "text": text_norm,
        "wall_labels": [int(x) for x in wall_labels],
        "pressure_level": [int(x) for x in pressure_level],
        "polarity": [int(x) for x in polarity],
        "source": str(source),
        "source_type": str(source_type),
        "source_trust": str(source_trust),
        "lang": "en",
        "split": split_norm,
        "label_quality": str(label_quality),
        "is_attack": int(is_attack),
    }
    record = _normalize_record(record)
    _validate_record(record)
    return record


def _split_rows(rows: Sequence[Dict[str, Any]], *, dev_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not rows:
        return [], []
    ratio = max(0.0, min(0.5, float(dev_ratio)))
    grouped: Dict[int, List[Dict[str, Any]]] = {0: [], 1: []}
    for row in rows:
        grouped[int(row.get("is_attack", 0))].append(dict(row))

    rng = random.Random(seed)
    train_rows: List[Dict[str, Any]] = []
    dev_rows: List[Dict[str, Any]] = []
    for label, items in grouped.items():
        rng.shuffle(items)
        dev_n = int(round(len(items) * ratio))
        dev_n = min(dev_n, max(1, len(items) - 1)) if len(items) > 2 else min(dev_n, len(items))
        dev_rows.extend(items[:dev_n])
        train_rows.extend(items[dev_n:])
        if not train_rows and items:
            train_rows.append(items[0])
            if items[0] in dev_rows:
                dev_rows.remove(items[0])

    rng.shuffle(train_rows)
    rng.shuffle(dev_rows)
    return train_rows, dev_rows


def _load_deepset_dataset(
    *,
    dataset_id: str,
    dataset_cfg: Mapping[str, Any],
    projector: Pi0IntentAwareV2,
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    root = str(dataset_cfg["path"])
    split_map = dataset_cfg.get("split_map", {}) or {}
    attack_label = int((dataset_cfg.get("label_mapping", {}) or {}).get("attack_value", 1))
    dev_ratio = float(dataset_cfg.get("dev_ratio", 0.2))
    ordinal_bins = list((dataset_cfg.get("labeling", {}) or {}).get("ordinal_bins", [0.45, 1.10, 2.00]))
    active_floor_gold = int((dataset_cfg.get("labeling", {}) or {}).get("active_floor_gold", 2))

    train_split = str(split_map.get("train", "train"))
    train_bundle = build_deepset_samples(
        benchmark_root=root,
        split=train_split,
        mode="full",
        max_samples=10_000_000,
        seed=seed,
        label_attack_value=attack_label,
    )
    raw_train: List[Dict[str, Any]] = []
    for sample in train_bundle.samples:
        raw_train.append(
            _build_record(
                sample_id=f"{dataset_id}:{sample.sample_id}",
                text=sample.text,
                source=dataset_id,
                source_type="user_input",
                source_trust="untrusted",
                split="train",
                is_attack=1 if sample.is_attack else 0,
                projector=projector,
                label_quality="silver",
                ordinal_bins=ordinal_bins,
                active_floor_gold=active_floor_gold,
            )
        )
    train_rows, dev_rows = _split_rows(raw_train, dev_ratio=dev_ratio, seed=seed)

    holdout_rows: List[Dict[str, Any]] = []
    holdout_split = split_map.get("holdout", split_map.get("test", "test"))
    holdout_bundle = build_deepset_samples(
        benchmark_root=root,
        split=str(holdout_split),
        mode="full",
        max_samples=10_000_000,
        seed=seed,
        label_attack_value=attack_label,
    )
    for sample in holdout_bundle.samples:
        holdout_rows.append(
            _build_record(
                sample_id=f"{dataset_id}:holdout:{sample.sample_id}",
                text=sample.text,
                source=f"{dataset_id}_holdout",
                source_type="user_input",
                source_trust="untrusted",
                split="holdout",
                is_attack=1 if sample.is_attack else 0,
                projector=projector,
                label_quality="gold",
                ordinal_bins=ordinal_bins,
                active_floor_gold=active_floor_gold,
            )
        )
    return {"train": train_rows, "dev": dev_rows, "holdout": holdout_rows}


def _load_wainject_dataset(
    *,
    dataset_id: str,
    dataset_cfg: Mapping[str, Any],
    projector: Pi0IntentAwareV2,
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    root = Path(str(dataset_cfg["path"]))
    dev_ratio = float(dataset_cfg.get("dev_ratio", 0.2))
    max_samples = int(dataset_cfg.get("max_samples", 0))
    ordinal_bins = list((dataset_cfg.get("labeling", {}) or {}).get("ordinal_bins", [0.45, 1.10, 2.00]))
    active_floor_gold = int((dataset_cfg.get("labeling", {}) or {}).get("active_floor_gold", 2))
    benign_limit = max_samples // 2 if max_samples > 0 else 0
    attack_limit = max_samples - benign_limit if max_samples > 0 else 0
    benign_paths = sorted((root / "benign").glob("*.jsonl"))
    malicious_paths = sorted((root / "malicious").glob("*.jsonl"))
    if not benign_paths or not malicious_paths:
        raise ValueError(f"{dataset_id} requires benign and malicious jsonl files under {root.as_posix()}")

    rows: List[Dict[str, Any]] = []
    benign_seen = 0
    attack_seen = 0
    for path in benign_paths:
        if benign_limit and benign_seen >= benign_limit:
            break
        for idx, row in enumerate(_iter_jsonl_rows(path), start=1):
            text = str(row.get("text", row.get("content", ""))).strip()
            if not text:
                continue
            rows.append(
                _build_record(
                    sample_id=f"{dataset_id}:{path.stem}:benign:{idx}",
                    text=text,
                    source=dataset_id,
                    source_type="web",
                    source_trust="untrusted",
                    split="train",
                    is_attack=0,
                    projector=projector,
                    label_quality="silver",
                    ordinal_bins=ordinal_bins,
                    active_floor_gold=active_floor_gold,
                )
            )
            benign_seen += 1
            if benign_limit and benign_seen >= benign_limit:
                break
    for path in malicious_paths:
        if attack_limit and attack_seen >= attack_limit:
            break
        for idx, row in enumerate(_iter_jsonl_rows(path), start=1):
            text = str(row.get("text", row.get("content", ""))).strip()
            if not text:
                continue
            rows.append(
                _build_record(
                    sample_id=f"{dataset_id}:{path.stem}:attack:{idx}",
                    text=text,
                    source=dataset_id,
                    source_type="web",
                    source_trust="untrusted",
                    split="train",
                    is_attack=1,
                    projector=projector,
                    label_quality="silver",
                    ordinal_bins=ordinal_bins,
                    active_floor_gold=active_floor_gold,
                )
            )
            attack_seen += 1
            if attack_limit and attack_seen >= attack_limit:
                break
    train_rows, dev_rows = _split_rows(rows, dev_ratio=dev_ratio, seed=seed)
    return {"train": train_rows, "dev": dev_rows, "holdout": []}


def _load_redteam_synth_dataset(
    *,
    dataset_id: str,
    dataset_cfg: Mapping[str, Any],
    projector: Pi0IntentAwareV2,
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    n_per_family = int(dataset_cfg.get("n_per_family", 200))
    dev_ratio = float(dataset_cfg.get("dev_ratio", 0.1))
    synth_seed = int(dataset_cfg.get("seed", seed))
    ordinal_bins = list((dataset_cfg.get("labeling", {}) or {}).get("ordinal_bins", [0.45, 1.10, 2.00]))
    active_floor_gold = int((dataset_cfg.get("labeling", {}) or {}).get("active_floor_gold", 2))
    rows: List[Dict[str, Any]] = []
    for idx, sample in enumerate(generate(seed=synth_seed, n_per_family=n_per_family), start=1):
        rows.append(
            _build_record(
                sample_id=f"{dataset_id}:{sample.id}:{idx}",
                text=sample.text,
                source=dataset_id,
                source_type="user_input",
                source_trust="untrusted",
                split="train",
                is_attack=1,
                projector=projector,
                label_quality="weak",
                ordinal_bins=ordinal_bins,
                active_floor_gold=active_floor_gold,
            )
        )
    train_rows, dev_rows = _split_rows(rows, dev_ratio=dev_ratio, seed=synth_seed)
    return {"train": train_rows, "dev": dev_rows, "holdout": []}


def _load_canonical_holdout(projector: Pi0IntentAwareV2) -> List[Dict[str, Any]]:
    holdout: List[Dict[str, Any]] = []
    for row in load_jsonl("tests/data/hard_negatives_50.jsonl"):
        holdout.append(
            _build_record(
                sample_id=f"hardneg:{row['id']}",
                text=str(row["text"]),
                source="hard_negatives",
                source_type="doc",
                source_trust="trusted",
                split="holdout",
                is_attack=0,
                projector=projector,
                label_quality="gold",
                ordinal_bins=[0.45, 1.10, 2.00],
                active_floor_gold=2,
            )
        )
    for row in load_jsonl("tests/data/redteam_pos_20.jsonl"):
        wall_labels = [1 if wall in set(row.get("expected_nonzero", [])) else 0 for wall in WALL_ORDER]
        polarity = [1 if wall_labels[i] else 0 for i in range(4)]
        pressure_level = [2 if wall_labels[i] else 0 for i in range(4)]
        holdout.append(
            {
                "sample_id": f"canonical_pos:{row['id']}",
                "text": str(row["text"]).strip(),
                "wall_labels": wall_labels,
                "pressure_level": pressure_level,
                "polarity": polarity,
                "source": "canonical_positive",
                "source_type": "user_input",
                "source_trust": "untrusted",
                "lang": "en",
                "split": "holdout",
                "label_quality": "gold",
                "is_attack": 1,
            }
        )
    for row in load_jsonl("tests/data/redteam_obf_20.jsonl"):
        wall_labels = [1 if wall in set(row.get("expected_nonzero", [])) else 0 for wall in WALL_ORDER]
        polarity = [1 if wall_labels[i] else 0 for i in range(4)]
        pressure_level = [2 if wall_labels[i] else 0 for i in range(4)]
        holdout.append(
            {
                "sample_id": f"canonical_obf:{row['id']}",
                "text": str(row["text"]).strip(),
                "wall_labels": wall_labels,
                "pressure_level": pressure_level,
                "polarity": polarity,
                "source": "canonical_obfuscated",
                "source_type": "user_input",
                "source_trust": "untrusted",
                "lang": "en",
                "split": "holdout",
                "label_quality": "gold",
                "is_attack": 1,
            }
        )

    for idx, sample in enumerate(generate(seed=19, n_per_family=240), start=1):
        wall_labels = [1 if wall in set(sample.targets) else 0 for wall in WALL_ORDER]
        polarity = [1 if wall_labels[i] else 0 for i in range(4)]
        pressure_level = [2 if wall_labels[i] else 0 for i in range(4)]
        holdout.append(
            {
                "sample_id": f"whitebox_holdout:{sample.id}:{idx}",
                "text": str(sample.text).strip(),
                "wall_labels": wall_labels,
                "pressure_level": pressure_level,
                "polarity": polarity,
                "source": "whitebox_holdout",
                "source_type": "user_input",
                "source_trust": "untrusted",
                "lang": "en",
                "split": "holdout",
                "label_quality": "gold",
                "is_attack": 1,
            }
        )

    for idx, row in enumerate(holdout):
        normalized = _normalize_record(row, split_override="holdout")
        _validate_record(normalized)
        holdout[idx] = normalized
    return holdout


def _mix_weighted(
    pools: Mapping[str, Sequence[Dict[str, Any]]],
    *,
    weights: Mapping[str, float],
    target_n: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    mutable: Dict[str, List[Dict[str, Any]]] = {k: [dict(x) for x in v] for k, v in pools.items() if v}
    if not mutable:
        return []
    rng = random.Random(seed)
    for values in mutable.values():
        rng.shuffle(values)
    max_total = sum(len(v) for v in mutable.values())
    target = max_total if target_n is None else max(1, min(max_total, int(target_n)))

    selected: List[Dict[str, Any]] = []
    while len(selected) < target and mutable:
        keys = sorted(mutable.keys())
        key_weights = [max(float(weights.get(k, 1.0)), 1e-6) for k in keys]
        chosen = rng.choices(keys, weights=key_weights, k=1)[0]
        bucket = mutable[chosen]
        selected.append(bucket.pop())
        if not bucket:
            del mutable[chosen]
    rng.shuffle(selected)
    return selected


def _split_stats(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    source_counts: Dict[str, int] = {}
    source_type_counts: Dict[str, int] = {}
    source_trust_counts: Dict[str, int] = {}
    chunk_bucket_counts: Dict[str, int] = {}
    quality_counts: Dict[str, int] = {}
    attack = 0
    benign = 0
    for row in rows:
        source = str(row.get("source", "unknown"))
        source_counts[source] = source_counts.get(source, 0) + 1
        source_type = str(row.get("source_type", "other"))
        source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1
        source_trust = str(row.get("source_trust", "untrusted"))
        source_trust_counts[source_trust] = source_trust_counts.get(source_trust, 0) + 1
        chunk_bucket = str(row.get("chunk_bucket", "64"))
        chunk_bucket_counts[chunk_bucket] = chunk_bucket_counts.get(chunk_bucket, 0) + 1
        quality = str(row.get("label_quality", "unknown"))
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
        if int(row.get("is_attack", 0)) == 1:
            attack += 1
        else:
            benign += 1
    return {
        "count": len(rows),
        "attack": attack,
        "benign": benign,
        "source_counts": dict(sorted(source_counts.items())),
        "source_type_counts": dict(sorted(source_type_counts.items())),
        "source_trust_counts": dict(sorted(source_trust_counts.items())),
        "chunk_bucket_counts": dict(sorted(chunk_bucket_counts.items())),
        "label_quality_counts": dict(sorted(quality_counts.items())),
    }


def _sample_ids_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    sample_ids = "\n".join(sorted(str(row["sample_id"]) for row in rows)).encode("utf-8")
    return _sha256_bytes(sample_ids)


def _records_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "\n".join(
        _stable_record_hash(row) for row in sorted(rows, key=lambda x: str(x["sample_id"]))
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _validate_license_policy(dataset_cfg: Mapping[str, Any]) -> None:
    license_policy = str(dataset_cfg.get("license_policy", "permissive")).lower()
    if license_policy != "permissive":
        raise ValueError("only permissive license_policy is allowed for pitheta train in this iteration")
    if not bool(dataset_cfg.get("allowed_for_train", False)):
        raise ValueError("dataset marked as not allowed_for_train")


def _resolve_content_filter_config(
    *,
    snapshot: Mapping[str, Any],
    output_dir: Path,
    content_filter: Optional[Mapping[str, Any]],
) -> DatasetContentFilter:
    cfg_from_resolved = ((snapshot.get("pitheta_train", {}) or {}).get("content_filter", {}) or {})
    merged: Dict[str, Any] = dict(cfg_from_resolved if isinstance(cfg_from_resolved, Mapping) else {})
    if isinstance(content_filter, Mapping):
        merged.update(dict(content_filter))

    env_mode = str(os.getenv("PITHETA_CONTENT_FILTER_MODE", "")).strip().lower()
    if env_mode:
        merged["mode"] = env_mode

    mode = str(merged.get("mode", CONTENT_FILTER_MODE_OFF)).strip().lower()
    if mode not in CONTENT_FILTER_MODES:
        raise ValueError(f"invalid content filter mode: {mode}")

    log_path_raw = str(merged.get("log_path", "")).strip()
    log_path: Optional[Path] = (output_dir / "content_filter_drops.jsonl") if (mode != CONTENT_FILTER_MODE_OFF) else None
    if log_path_raw:
        candidate = Path(log_path_raw)
        if not candidate.is_absolute():
            candidate = (output_dir / candidate).resolve()
        log_path = candidate

    return DatasetContentFilter(
        mode=mode,
        fail_closed=bool(merged.get("fail_closed", False)),
        api_key_env=str(merged.get("api_key_env", "OPENAI_API_KEY")),
        base_url=str(merged.get("base_url", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))),
        model=str(merged.get("model", "omni-moderation-latest")),
        timeout_sec=float(merged.get("timeout_sec", 20.0)),
        max_retries=int(merged.get("max_retries", 2)),
        backoff_sec=float(merged.get("backoff_sec", 0.75)),
        block_categories=tuple(merged.get("block_categories", OPENAI_MODERATION_BLOCK_CATEGORIES_DEFAULT)),
        block_score_threshold=float(merged.get("block_score_threshold", 0.0)),
        apply_splits=tuple(merged.get("apply_splits", ("train", "dev", "holdout"))),
        log_path=log_path,
    )


def _apply_content_filter_to_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    split: str,
    content_filter: DatasetContentFilter,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if (not content_filter.active) or (not content_filter.should_apply_to_split(split)):
        return [dict(r) for r in rows], {"checked": 0, "dropped": 0, "errors": 0, "drop_reasons": {}}

    kept: List[Dict[str, Any]] = []
    checked = 0
    dropped = 0
    errors = 0
    drop_reasons: Dict[str, int] = {}

    for row in rows:
        row_dict = dict(row)
        text = str(row_dict.get("text", ""))
        decision = content_filter.decide(text)
        checked += 1
        if decision.error:
            errors += 1
        if decision.blocked:
            dropped += 1
            reason = str(decision.reason or "blocked")
            drop_reasons[reason] = int(drop_reasons.get(reason, 0)) + 1
            if content_filter.log_path is not None:
                _append_jsonl_row(
                    content_filter.log_path,
                    {
                        "sample_id": str(row_dict.get("sample_id", "")),
                        "split": str(split),
                        "source": str(row_dict.get("source", "")),
                        "provider": str(decision.provider),
                        "reason": reason,
                        "categories": list(decision.categories),
                        "error": str(decision.error or ""),
                    },
                )
            continue
        kept.append(row_dict)

    return kept, {
        "checked": int(checked),
        "dropped": int(dropped),
        "errors": int(errors),
        "drop_reasons": dict(sorted(drop_reasons.items())),
    }


def _build_manifests(
    *,
    output_dir: Path,
    registry_path: str,
    registry_cfg: Mapping[str, Any],
    config_sha: str,
    seed: int,
    rows_train: Sequence[Mapping[str, Any]],
    rows_dev: Sequence[Mapping[str, Any]],
    rows_holdout: Sequence[Mapping[str, Any]],
    strict: bool,
    content_filter_summary: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    registry_hash = _sha256_file(Path(registry_path))
    schema_hash = _sha256_bytes(json.dumps({"schema_version": PI_THETA_SCHEMA_VERSION, "walls": WALL_ORDER}).encode("utf-8"))

    split_sample_ids_sha = {
        "train": _sample_ids_sha256(rows_train),
        "dev": _sample_ids_sha256(rows_dev),
        "holdout": _sample_ids_sha256(rows_holdout),
    }
    split_records_sha = {
        "train": _records_sha256(rows_train),
        "dev": _records_sha256(rows_dev),
        "holdout": _records_sha256(rows_holdout),
    }

    dataset_manifest = {
        "schema_version": PI_THETA_SCHEMA_VERSION,
        "registry_path": Path(registry_path).as_posix(),
        "registry_sha256": registry_hash,
        "resolved_config_sha256": config_sha,
        "seed": int(seed),
        "sampling": dict((registry_cfg.get("sampling", {}) or {})),
        "splits": {
            "train": _split_stats(rows_train),
            "dev": _split_stats(rows_dev),
            "holdout": _split_stats(rows_holdout),
        },
        "sample_ids_sha256": split_sample_ids_sha,
        "records_sha256": split_records_sha,
        "schema_hash": schema_hash,
    }
    if isinstance(content_filter_summary, Mapping):
        dataset_manifest["content_filter"] = dict(content_filter_summary)
    holdout_manifest = {
        "schema_version": PI_THETA_SCHEMA_VERSION,
        "seed": int(seed),
        "schema_hash": schema_hash,
        "sample_ids": sorted(str(row["sample_id"]) for row in rows_holdout),
        "sample_ids_sha256": split_sample_ids_sha["holdout"],
        "records_sha256": split_records_sha["holdout"],
        "split_stats": _split_stats(rows_holdout),
    }

    dataset_path = output_dir / "dataset_manifest.json"
    if strict and dataset_path.exists():
        existing = json.loads(dataset_path.read_text(encoding="utf-8"))
        existing_seed = existing.get("seed")
        if existing_seed is not None and int(existing_seed) != int(seed):
            raise ValueError(
                f"strict dataset immutability violated: seed changed ({existing_seed} -> {seed})"
            )

        existing_sample_map = existing.get("sample_ids_sha256", {}) or {}
        if not isinstance(existing_sample_map, dict):
            existing_sample_map = {}
        existing_record_map = existing.get("records_sha256", {}) or {}
        if not isinstance(existing_record_map, dict):
            existing_record_map = {}

        for split in ("train", "dev", "holdout"):
            prev_sid = existing_sample_map.get(split)
            if prev_sid is not None and str(prev_sid) != str(split_sample_ids_sha[split]):
                raise ValueError(
                    f"strict dataset immutability violated: {split} sample_ids changed"
                )
            prev_rec = existing_record_map.get(split)
            if prev_rec is not None and str(prev_rec) != str(split_records_sha[split]):
                raise ValueError(
                    f"strict dataset immutability violated: {split} records changed"
                )

    holdout_path = output_dir / "holdout_manifest.json"
    if strict and holdout_path.exists():
        existing = json.loads(holdout_path.read_text(encoding="utf-8"))
        if existing.get("sample_ids_sha256") != holdout_manifest["sample_ids_sha256"]:
            raise ValueError("strict holdout immutability violated: holdout sample_ids changed")
        if existing.get("records_sha256") != holdout_manifest["records_sha256"]:
            raise ValueError("strict holdout immutability violated: holdout records changed")

    (output_dir / "dataset_manifest.json").write_text(json.dumps(dataset_manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    (output_dir / "holdout_manifest.json").write_text(json.dumps(holdout_manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    (output_dir / "config_hashes.json").write_text(
        json.dumps(
            {
                "resolved_config_sha256": config_sha,
                "registry_sha256": registry_hash,
                "dataset_manifest_sha256": _sha256_file(output_dir / "dataset_manifest.json"),
                "holdout_manifest_sha256": _sha256_file(output_dir / "holdout_manifest.json"),
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"dataset_manifest": dataset_manifest, "holdout_manifest": holdout_manifest}


def build_pitheta_dataset_artifacts(
    *,
    registry_path: str,
    output_dir: str,
    seed: int = 41,
    profile: str = "dev",
    strict: bool = False,
    use_semantic_labeling: bool = False,
    content_filter: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    registry = _load_registry(registry_path)
    snapshot = load_resolved_config(
        profile=profile,
        cli_overrides=(
            None
            if use_semantic_labeling
            else {
                "pi0": {
                    "semantic": {
                        "enabled": "false",
                    }
                }
            }
        ),
    )
    projector = Pi0IntentAwareV2(snapshot.resolved)
    labeling_cfg = (snapshot.resolved.get("pitheta_train", {}) or {}).get("labeling", {}) or {}
    ordinal_bins = list(labeling_cfg.get("ordinal_bins", [0.45, 1.10, 2.00]))
    active_floor_gold = int(labeling_cfg.get("active_floor_gold", 2))
    content_filter_engine = _resolve_content_filter_config(
        snapshot=snapshot.resolved,
        output_dir=output,
        content_filter=content_filter,
    )

    dataset_rows: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for ds_cfg_raw in registry.get("datasets", []):
        if not isinstance(ds_cfg_raw, MutableMapping):
            raise ValueError("dataset entries in registry must be mappings")
        ds_cfg = dict(ds_cfg_raw)
        ds_id = str(ds_cfg.get("dataset_id", "")).strip()
        if not ds_id:
            raise ValueError("dataset_id is required")
        if strict:
            _validate_license_policy(ds_cfg)
        if not bool(ds_cfg.get("allowed_for_train", False)):
            continue
        loader = str(ds_cfg.get("loader", "")).strip().lower()
        if loader == "deepset":
            ds_cfg = {
                **ds_cfg,
                "labeling": {
                    **(ds_cfg.get("labeling", {}) or {}),
                    "ordinal_bins": ordinal_bins,
                    "active_floor_gold": active_floor_gold,
                },
            }
            dataset_rows[ds_id] = _load_deepset_dataset(dataset_id=ds_id, dataset_cfg=ds_cfg, projector=projector, seed=seed)
        elif loader == "wainject_text":
            ds_cfg = {
                **ds_cfg,
                "labeling": {
                    **(ds_cfg.get("labeling", {}) or {}),
                    "ordinal_bins": ordinal_bins,
                    "active_floor_gold": active_floor_gold,
                },
            }
            dataset_rows[ds_id] = _load_wainject_dataset(dataset_id=ds_id, dataset_cfg=ds_cfg, projector=projector, seed=seed)
        elif loader == "redteam_synth":
            ds_cfg = {
                **ds_cfg,
                "labeling": {
                    **(ds_cfg.get("labeling", {}) or {}),
                    "ordinal_bins": ordinal_bins,
                    "active_floor_gold": active_floor_gold,
                },
            }
            dataset_rows[ds_id] = _load_redteam_synth_dataset(dataset_id=ds_id, dataset_cfg=ds_cfg, projector=projector, seed=seed)
        else:
            raise ValueError(f"unsupported dataset loader for {ds_id}: {loader}")

    sampling_cfg = registry.get("sampling", {}) or {}
    dataset_weights: Dict[str, float] = {}
    train_pools: Dict[str, List[Dict[str, Any]]] = {}
    dev_pools: Dict[str, List[Dict[str, Any]]] = {}
    holdout_rows: List[Dict[str, Any]] = []
    for ds_id, bundles in dataset_rows.items():
        ds_cfg = next((d for d in registry.get("datasets", []) if d.get("dataset_id") == ds_id), {})
        dataset_weights[ds_id] = float(ds_cfg.get("sampling_weight", 1.0))
        train_pools[ds_id] = list(bundles.get("train", []))
        dev_pools[ds_id] = list(bundles.get("dev", []))
        holdout_rows.extend(bundles.get("holdout", []))

    train_rows = _mix_weighted(
        train_pools,
        weights=dataset_weights,
        target_n=int(sampling_cfg["target_train_samples"]) if "target_train_samples" in sampling_cfg else None,
        seed=seed,
    )
    dev_rows = _mix_weighted(
        dev_pools,
        weights=dataset_weights,
        target_n=int(sampling_cfg["target_dev_samples"]) if "target_dev_samples" in sampling_cfg else None,
        seed=seed + 17,
    )
    holdout_rows.extend(_load_canonical_holdout(projector))

    # Deduplicate by sample_id while preserving deterministic order.
    seen: set[str] = set()
    deduped_holdout: List[Dict[str, Any]] = []
    for row in sorted(holdout_rows, key=lambda x: str(x["sample_id"])):
        sid = str(row["sample_id"])
        if sid in seen:
            continue
        seen.add(sid)
        row_copy = _normalize_record(row, split_override="holdout")
        _validate_record(row_copy)
        deduped_holdout.append(row_copy)
    holdout_rows = deduped_holdout

    for idx, row in enumerate(train_rows):
        row_norm = _normalize_record(row, split_override="train")
        _validate_record(row_norm)
        train_rows[idx] = row_norm
    for idx, row in enumerate(dev_rows):
        row_norm = _normalize_record(row, split_override="dev")
        _validate_record(row_norm)
        dev_rows[idx] = row_norm

    train_rows, train_filter_stats = _apply_content_filter_to_rows(
        rows=train_rows,
        split="train",
        content_filter=content_filter_engine,
    )
    dev_rows, dev_filter_stats = _apply_content_filter_to_rows(
        rows=dev_rows,
        split="dev",
        content_filter=content_filter_engine,
    )
    holdout_rows, holdout_filter_stats = _apply_content_filter_to_rows(
        rows=holdout_rows,
        split="holdout",
        content_filter=content_filter_engine,
    )
    content_filter_summary = {
        "mode": content_filter_engine.mode,
        "fail_closed": bool(content_filter_engine.fail_closed),
        "apply_splits": list(content_filter_engine.apply_splits),
        "stats": {
            "train": train_filter_stats,
            "dev": dev_filter_stats,
            "holdout": holdout_filter_stats,
            "dropped_total": int(
                int(train_filter_stats.get("dropped", 0))
                + int(dev_filter_stats.get("dropped", 0))
                + int(holdout_filter_stats.get("dropped", 0))
            ),
        },
        "log_path": (content_filter_engine.log_path.as_posix() if content_filter_engine.log_path is not None else None),
    }

    _jsonl_write(output / "train.jsonl", train_rows)
    _jsonl_write(output / "dev.jsonl", dev_rows)
    _jsonl_write(output / "holdout.jsonl", holdout_rows)

    (output / "sample_ids_train.txt").write_text(
        "\n".join(sorted(str(r["sample_id"]) for r in train_rows)) + ("\n" if train_rows else ""),
        encoding="utf-8",
    )
    (output / "sample_ids_dev.txt").write_text(
        "\n".join(sorted(str(r["sample_id"]) for r in dev_rows)) + ("\n" if dev_rows else ""),
        encoding="utf-8",
    )
    (output / "sample_ids_holdout.txt").write_text(
        "\n".join(sorted(str(r["sample_id"]) for r in holdout_rows)) + ("\n" if holdout_rows else ""),
        encoding="utf-8",
    )

    manifests = _build_manifests(
        output_dir=output,
        registry_path=registry_path,
        registry_cfg=registry,
        config_sha=snapshot.resolved_sha256,
        seed=seed,
        rows_train=train_rows,
        rows_dev=dev_rows,
        rows_holdout=holdout_rows,
        strict=strict,
        content_filter_summary=content_filter_summary,
    )

    report = {
        "status": "ok",
        "schema_version": PI_THETA_SCHEMA_VERSION,
        "registry_path": Path(registry_path).as_posix(),
        "output_dir": output.as_posix(),
        "profile": profile,
        "seed": int(seed),
        "resolved_config_sha256": snapshot.resolved_sha256,
        "counts": {
            "train": len(train_rows),
            "dev": len(dev_rows),
            "holdout": len(holdout_rows),
        },
        "manifests": {
            "dataset_manifest": (output / "dataset_manifest.json").as_posix(),
            "holdout_manifest": (output / "holdout_manifest.json").as_posix(),
            "config_hashes": (output / "config_hashes.json").as_posix(),
        },
        "content_filter": content_filter_summary,
        "fingerprints": {
            "train_sha256": _sha256_file(output / "train.jsonl"),
            "dev_sha256": _sha256_file(output / "dev.jsonl"),
            "holdout_sha256": _sha256_file(output / "holdout.jsonl"),
            "sample_rows_sha256": _sha256_bytes(
                "\n".join(
                    _stable_record_hash(r) for r in sorted(train_rows + dev_rows + holdout_rows, key=lambda x: str(x["sample_id"]))
                ).encode("utf-8")
            ),
        },
        "manifest_summary": manifests,
    }
    # Keep CLI output small and deterministic for automation logs.
    report["manifest_summary"] = {
        "dataset_manifest": {
            "schema_version": manifests["dataset_manifest"].get("schema_version"),
            "seed": manifests["dataset_manifest"].get("seed"),
            "splits": manifests["dataset_manifest"].get("splits", {}),
            "sample_ids_sha256": manifests["dataset_manifest"].get("sample_ids_sha256", {}),
        },
        "holdout_manifest": {
            "sample_ids_sha256": manifests["holdout_manifest"].get("sample_ids_sha256"),
            "records_sha256": manifests["holdout_manifest"].get("records_sha256"),
            "count": int((manifests["holdout_manifest"].get("split_stats", {}) or {}).get("count", 0)),
        },
    }
    (output / "build_report.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    return report


def load_pitheta_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = _jsonl_read(Path(path))
    normalized_rows: List[Dict[str, Any]] = []
    for row in rows:
        row_norm = _normalize_record(row)
        _validate_record(row_norm)
        normalized_rows.append(row_norm)
    return normalized_rows
