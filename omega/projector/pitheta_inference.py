"""Inference runtime for PiTheta projector."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from omega.interfaces.contracts_v1 import WALLS_V1
from omega.pitheta.calibration import apply_temperature

WALLS = list(WALLS_V1)


@dataclass(frozen=True)
class PiThetaInferenceConfig:
    checkpoint_dir: str
    base_model_path: str
    max_length: int = 256
    batch_size: int = 8
    device: str = "auto"
    local_files_only: bool = True
    head_mode: str = "auto"
    calibration_file: str = "temperature_scaling.json"
    require_calibration: bool = True


class PiThetaInference:
    """Loads encoder + LoRA adapter + PiTheta heads and runs predictions."""

    def __init__(self, config: PiThetaInferenceConfig) -> None:
        self.config = config
        self._torch = None
        self._tokenizer = None
        self._encoder = None
        self._wall_head = None
        self._ordinal_head = None
        self._polarity_head = None
        self._device = None
        self._runtime_error: Optional[str] = None
        self._meta: Dict[str, Any] = {}
        self._ready = False
        self._head_version = "unknown"
        self._temperature_payload: Dict[str, Any] = {}
        self._calibration_active = False
        self._init_runtime()

    @property
    def ready(self) -> bool:
        return bool(self._ready)

    @property
    def runtime_error(self) -> Optional[str]:
        return self._runtime_error

    @property
    def metadata(self) -> Dict[str, Any]:
        base = dict(self._meta)
        base["head_version"] = self._head_version
        base["calibration_active"] = bool(self._calibration_active)
        return base

    @property
    def head_version(self) -> str:
        return str(self._head_version)

    @property
    def calibration_active(self) -> bool:
        return bool(self._calibration_active)

    @property
    def calibration_payload(self) -> Dict[str, Any]:
        return dict(self._temperature_payload)

    def _resolve_device(self, torch_mod, requested: str):
        if requested != "auto":
            return torch_mod.device(requested)
        if torch_mod.cuda.is_available():
            return torch_mod.device("cuda")
        return torch_mod.device("cpu")

    def _load_tokenizer(self, model_path: str):
        errors: List[str] = []
        for use_fast in (True, False):
            try:
                from transformers import AutoTokenizer

                return AutoTokenizer.from_pretrained(
                    model_path,
                    local_files_only=bool(self.config.local_files_only),
                    use_fast=use_fast,
                )
            except Exception as exc:
                errors.append(f"use_fast={use_fast}: {exc}")
        try:
            from transformers import DebertaV2Tokenizer

            return DebertaV2Tokenizer.from_pretrained(
                model_path,
                local_files_only=bool(self.config.local_files_only),
            )
        except Exception as exc:
            errors.append(f"DebertaV2Tokenizer fallback: {exc}")
        raise RuntimeError(
            "failed to load tokenizer; install sentencepiece and validate local model files. "
            + " | ".join(errors)
        )

    def _load_manifest(self, checkpoint_dir: Path) -> Dict[str, Any]:
        manifest_path = checkpoint_dir / "model_manifest.json"
        if manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "base_model": self.config.base_model_path,
            "num_walls": 4,
            "walls": WALLS,
            "head_version": "legacy_v1",
        }

    def _detect_head_version(self, payload: Dict[str, Any], manifest: Dict[str, Any]) -> str:
        configured = str(self.config.head_mode or "auto").lower()
        if configured not in {"auto", "legacy", "ordinal"}:
            raise RuntimeError(f"unsupported head_mode={configured}")
        manifest_head = str(manifest.get("head_version", "auto")).lower()
        has_ordinal = bool(payload.get("ordinal_head_state_dict"))
        has_wall = bool(payload.get("wall_head_state_dict"))
        if configured == "legacy":
            if not has_wall:
                raise RuntimeError("legacy head_mode requested but wall_head_state_dict missing")
            return "legacy_v1"
        if configured == "ordinal":
            if not has_ordinal:
                raise RuntimeError("ordinal head_mode requested but ordinal_head_state_dict missing")
            return "ordinal_v2"
        if has_ordinal:
            return "ordinal_v2"
        if has_wall:
            return "legacy_v1"
        if manifest_head in {"legacy_v1", "ordinal_v2"}:
            return manifest_head
        raise RuntimeError("unable to infer checkpoint head version")

    def _load_temperature_payload(self, checkpoint_dir: Path) -> Dict[str, Any]:
        rel = str(self.config.calibration_file or "temperature_scaling.json")
        path = Path(rel)
        if not path.is_absolute():
            path = checkpoint_dir / path
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return {}
            return payload
        except Exception:
            return {}

    def _init_runtime(self) -> None:
        ckpt = Path(self.config.checkpoint_dir)
        if not ckpt.exists():
            self._runtime_error = f"checkpoint_dir_not_found:{ckpt.as_posix()}"
            return

        try:
            import torch
            from transformers import AutoModel
        except Exception as exc:  # pragma: no cover - dependency gate
            self._runtime_error = f"missing_runtime_dependencies:{exc}"
            return

        self._torch = torch
        self._device = self._resolve_device(torch, self.config.device)
        manifest = self._load_manifest(ckpt)
        base_model = str(manifest.get("base_model", self.config.base_model_path))
        tokenizer_path = ckpt / "tokenizer"
        tokenizer_model = tokenizer_path if tokenizer_path.exists() else Path(base_model)

        try:
            tokenizer = self._load_tokenizer(str(tokenizer_model))
            encoder = AutoModel.from_pretrained(
                str(base_model),
                local_files_only=bool(self.config.local_files_only),
            )
        except Exception as exc:  # pragma: no cover - runtime file mismatch
            self._runtime_error = f"model_load_failed:{exc}"
            return

        adapter_dir = ckpt / "adapter"
        if adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
            try:
                from peft import PeftModel

                encoder = PeftModel.from_pretrained(encoder, str(adapter_dir), is_trainable=False)
            except Exception as exc:  # pragma: no cover - optional dependency
                self._runtime_error = f"adapter_load_failed:{exc}"
                return

        hidden_size = int(getattr(encoder.config, "hidden_size"))
        wall_head = torch.nn.Linear(hidden_size, 4)
        ordinal_head = torch.nn.Linear(hidden_size, 4 * 4)
        polarity_head = torch.nn.Linear(hidden_size, 4 * 3)

        heads_path = ckpt / "heads.pt"
        if not heads_path.exists():
            self._runtime_error = f"heads_not_found:{heads_path.as_posix()}"
            return

        try:
            payload = torch.load(str(heads_path), map_location="cpu")
            self._head_version = self._detect_head_version(payload=payload, manifest=manifest)
            pol_state = payload.get("polarity_head_state_dict", {})
            polarity_head.load_state_dict(pol_state, strict=True)
            if self._head_version == "legacy_v1":
                wall_state = payload.get("wall_head_state_dict", {})
                wall_head.load_state_dict(wall_state, strict=True)
            else:
                ord_state = payload.get("ordinal_head_state_dict", {})
                ordinal_head.load_state_dict(ord_state, strict=True)
        except Exception as exc:  # pragma: no cover - runtime file mismatch
            self._runtime_error = f"heads_load_failed:{exc}"
            return

        temperature = self._load_temperature_payload(ckpt)
        has_ord_temp = isinstance((temperature.get("ordinal", {}) or {}).get("temperatures", None), list)
        has_pol_temp = isinstance((temperature.get("polarity", {}) or {}).get("temperatures", None), list)
        self._temperature_payload = temperature
        self._calibration_active = bool(has_pol_temp and (has_ord_temp or self._head_version == "legacy_v1"))
        if bool(self.config.require_calibration) and not self._calibration_active and self._head_version == "ordinal_v2":
            self._runtime_error = "calibration_required_but_missing"
            return

        encoder.eval()
        wall_head.eval()
        ordinal_head.eval()
        polarity_head.eval()
        encoder.to(self._device)
        wall_head.to(self._device)
        ordinal_head.to(self._device)
        polarity_head.to(self._device)

        self._tokenizer = tokenizer
        self._encoder = encoder
        self._wall_head = wall_head
        self._ordinal_head = ordinal_head
        self._polarity_head = polarity_head
        self._meta = {
            "base_model": base_model,
            "checkpoint_dir": ckpt.as_posix(),
            "device": str(self._device),
            "has_adapter": bool(adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists()),
        }
        self._ready = True
        self._runtime_error = None

    def _mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        return summed / denom

    def _temperature(self, key: str, default_len: int) -> List[float]:
        block = self._temperature_payload.get(key, {}) if isinstance(self._temperature_payload, dict) else {}
        vals = block.get("temperatures", []) if isinstance(block, dict) else []
        if not isinstance(vals, list) or len(vals) != default_len:
            return [1.0] * default_len
        return [max(float(x), 1e-6) for x in vals]

    def predict_outputs(self, texts: Iterable[str]) -> Dict[str, Any]:
        if not self._ready:
            raise RuntimeError(self._runtime_error or "pitheta runtime is not ready")
        rows = [str(x) for x in texts]
        if not rows:
            return {
                "head_version": self._head_version,
                "calibrated": bool(self._calibration_active),
                "wall_prob": np.zeros((0, 4), dtype=np.float32),
                "ordinal_prob": np.zeros((0, 4, 4), dtype=np.float32),
                "polarity_prob": np.zeros((0, 4, 3), dtype=np.float32),
            }

        torch_mod = self._torch
        wall_out: List[np.ndarray] = []
        ord_out: List[np.ndarray] = []
        pol_out: List[np.ndarray] = []
        bs = max(1, int(self.config.batch_size))
        with torch_mod.no_grad():
            for i in range(0, len(rows), bs):
                batch = rows[i : i + bs]
                tokens = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=int(self.config.max_length),
                    return_tensors="pt",
                )
                tokens = {k: v.to(self._device) for k, v in tokens.items()}
                encoded = self._encoder(**tokens)
                pooled = self._mean_pool(encoded.last_hidden_state, tokens["attention_mask"])
                pol_logits = self._polarity_head(pooled).reshape((-1, 4, 3))
                pol_logits_np = pol_logits.detach().cpu().numpy().astype(np.float32)
                if self._calibration_active:
                    pol_logits_np = apply_temperature(pol_logits_np, self._temperature("polarity", 4))
                pol_prob = torch_mod.softmax(torch_mod.tensor(pol_logits_np), dim=-1).detach().cpu().numpy().astype(np.float32)
                pol_out.append(pol_prob)

                if self._head_version == "legacy_v1":
                    wall_logits = self._wall_head(pooled)
                    wall_prob = torch_mod.sigmoid(wall_logits).detach().cpu().numpy().astype(np.float32)
                    wall_out.append(wall_prob)
                else:
                    ord_logits = self._ordinal_head(pooled).reshape((-1, 4, 4))
                    ord_logits_np = ord_logits.detach().cpu().numpy().astype(np.float32)
                    if self._calibration_active:
                        ord_logits_np = apply_temperature(ord_logits_np, self._temperature("ordinal", 4))
                    ord_prob = torch_mod.softmax(torch_mod.tensor(ord_logits_np), dim=-1).detach().cpu().numpy().astype(np.float32)
                    ord_out.append(ord_prob)

        return {
            "head_version": self._head_version,
            "calibrated": bool(self._calibration_active),
            "wall_prob": np.vstack(wall_out) if wall_out else np.zeros((len(rows), 4), dtype=np.float32),
            "ordinal_prob": np.vstack(ord_out) if ord_out else np.zeros((len(rows), 4, 4), dtype=np.float32),
            "polarity_prob": np.vstack(pol_out) if pol_out else np.zeros((len(rows), 4, 3), dtype=np.float32),
        }

    def predict(self, texts: Iterable[str]):
        """Backward-compatible output used by old callers."""
        out = self.predict_outputs(texts)
        return out["wall_prob"], out["polarity_prob"]

