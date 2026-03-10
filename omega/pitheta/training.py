"""Training utilities for PiTheta LoRA."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import yaml

from omega.pitheta.calibration import build_temperature_payload
from omega.pitheta.dataset_builder import load_pitheta_jsonl

PRESSURE_MAP_DEFAULT = [0.0, 0.25, 0.6, 1.0]


def _optional_imports():
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(f"missing training dependencies torch/transformers: {exc}") from exc
    return torch, AutoModel, AutoTokenizer


def _load_tokenizer(base_model: str, *, local_files_only: bool):
    _, _, AutoTokenizer = _optional_imports()
    errors: List[str] = []
    for use_fast in (True, False):
        try:
            return AutoTokenizer.from_pretrained(
                base_model,
                local_files_only=bool(local_files_only),
                use_fast=use_fast,
            )
        except Exception as exc:
            errors.append(f"use_fast={use_fast}: {exc}")
    try:
        from transformers import DebertaV2Tokenizer

        return DebertaV2Tokenizer.from_pretrained(
            base_model,
            local_files_only=bool(local_files_only),
        )
    except Exception as exc:
        errors.append(f"DebertaV2Tokenizer fallback: {exc}")
    raise RuntimeError(
        "failed to load tokenizer; install sentencepiece and validate local model files. "
        + " | ".join(errors)
    )


def _read_weight_grid(raw: Any, *, classes: int) -> List[List[float]]:
    if raw is None:
        return [[1.0] * classes for _ in range(4)]
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError(f"class_weights must be a list of 4 lists (classes={classes})")
    grid: List[List[float]] = []
    for row in raw:
        if not isinstance(row, list) or len(row) != classes:
            raise ValueError(f"class_weights row must have {classes} entries")
        vals = [float(x) for x in row]
        if any(v <= 0 for v in vals):
            raise ValueError("class_weights entries must be > 0")
        grid.append(vals)
    return grid


@dataclass(frozen=True)
class TrainConfig:
    base_model: str
    task: str
    max_len: int
    batch_size: int
    grad_accum_steps: int
    lr: float
    epochs: int
    warmup_ratio: float
    weight_decay: float
    seed: int
    mixed_precision: str
    early_stopping_patience: int
    early_stopping_metric: str
    benign_fpr_ceiling: float
    lora_enabled: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: List[str]
    loss_weight_ordinal: float
    loss_weight_polarity: float
    ordinal_class_weights: List[List[float]]
    polarity_class_weights: List[List[float]]
    fit_temperature: bool
    temperature_split: str
    temperature_output: str
    calibration_source_mode: str
    calibration_gold_slice_path: str
    calibration_gold_ratio: float
    calibration_weak_ratio: float
    pressure_map: List[float]

    @property
    def use_ordinal(self) -> bool:
        return str(self.task).strip().lower() == "ordinal_pressure_with_polarity"


def load_train_config(path: str) -> TrainConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "pitheta_train" in raw:
        raw = raw["pitheta_train"]
    if not isinstance(raw, dict):
        raise ValueError("pitheta_train config must be mapping")
    lora = raw.get("lora", {}) or {}
    loss_weights = raw.get("loss_weights", {}) or {}
    ordinal_cfg = raw.get("ordinal", {}) or {}
    polarity_cfg = raw.get("polarity", {}) or {}
    calibration_cfg = raw.get("calibration", {}) or {}
    cfg = TrainConfig(
        base_model=str(raw.get("base_model", "microsoft/deberta-v3-base")),
        task=str(raw.get("task", "ordinal_pressure_with_polarity")),
        max_len=int(raw.get("max_len", 256)),
        batch_size=int(raw.get("batch_size", 32)),
        grad_accum_steps=int(raw.get("grad_accum_steps", 1)),
        lr=float(raw.get("lr", 2e-4)),
        epochs=int(raw.get("epochs", 3)),
        warmup_ratio=float(raw.get("warmup_ratio", 0.1)),
        weight_decay=float(raw.get("weight_decay", 0.01)),
        seed=int(raw.get("seed", 41)),
        mixed_precision=str(raw.get("mixed_precision", "auto")),
        early_stopping_patience=int(raw.get("early_stopping_patience", 2)),
        early_stopping_metric=str(raw.get("early_stopping_metric", "dev_attack_recall")),
        benign_fpr_ceiling=float(raw.get("benign_fpr_ceiling", 0.05)),
        lora_enabled=bool(lora.get("enabled", True)),
        lora_r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        lora_target_modules=[str(x) for x in list(lora.get("target_modules", ["query_proj", "value_proj"]))],
        loss_weight_ordinal=float(loss_weights.get("ordinal", 1.0)),
        loss_weight_polarity=float(loss_weights.get("polarity", 0.3)),
        ordinal_class_weights=_read_weight_grid(ordinal_cfg.get("class_weights", None), classes=4),
        polarity_class_weights=_read_weight_grid(polarity_cfg.get("class_weights", None), classes=3),
        fit_temperature=bool(calibration_cfg.get("fit_temperature", True)),
        temperature_split=str(calibration_cfg.get("temperature_split", "dev")).lower(),
        temperature_output=str(calibration_cfg.get("temperature_output", "best/temperature_scaling.json")),
        calibration_source_mode=str(calibration_cfg.get("source_mode", "dataset")).lower(),
        calibration_gold_slice_path=str(calibration_cfg.get("gold_slice_path", "data/gold_slice/gold_slice.jsonl")),
        calibration_gold_ratio=float(calibration_cfg.get("gold_ratio", 0.70)),
        calibration_weak_ratio=float(calibration_cfg.get("weak_ratio", 0.30)),
        pressure_map=[float(x) for x in list(raw.get("pressure_map", PRESSURE_MAP_DEFAULT))],
    )
    validate_train_config(cfg)
    return cfg


def validate_train_config(cfg: TrainConfig) -> None:
    if cfg.max_len <= 0:
        raise ValueError("max_len must be > 0")
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if cfg.grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be > 0")
    if cfg.lr <= 0:
        raise ValueError("lr must be > 0")
    if cfg.epochs <= 0:
        raise ValueError("epochs must be > 0")
    if cfg.lora_r <= 0:
        raise ValueError("lora.r must be > 0")
    if cfg.lora_alpha <= 0:
        raise ValueError("lora.alpha must be > 0")
    if cfg.lora_dropout < 0:
        raise ValueError("lora.dropout must be >= 0")
    if cfg.loss_weight_ordinal <= 0:
        raise ValueError("loss_weights.ordinal must be > 0")
    if cfg.loss_weight_polarity < 0:
        raise ValueError("loss_weights.polarity must be >= 0")
    if cfg.temperature_split not in {"dev", "holdout", "calibration"}:
        raise ValueError("temperature_split must be dev|holdout|calibration")
    if cfg.calibration_source_mode not in {"dataset", "gold_only", "blended"}:
        raise ValueError("calibration.source_mode must be dataset|gold_only|blended")
    if not str(cfg.calibration_gold_slice_path).strip():
        raise ValueError("calibration.gold_slice_path must be non-empty")
    if cfg.calibration_gold_ratio < 0 or cfg.calibration_gold_ratio > 1:
        raise ValueError("calibration.gold_ratio must be in [0,1]")
    if cfg.calibration_weak_ratio < 0 or cfg.calibration_weak_ratio > 1:
        raise ValueError("calibration.weak_ratio must be in [0,1]")
    ratio_sum = cfg.calibration_gold_ratio + cfg.calibration_weak_ratio
    if cfg.calibration_source_mode == "blended" and abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("for blended calibration, gold_ratio + weak_ratio must equal 1.0")
    if not str(cfg.temperature_output).strip():
        raise ValueError("temperature_output must be non-empty")
    if len(cfg.pressure_map) != 4:
        raise ValueError("pressure_map must have 4 values")
    last = -1.0
    for value in cfg.pressure_map:
        if float(value) < 0:
            raise ValueError("pressure_map values must be >= 0")
        if float(value) < last:
            raise ValueError("pressure_map must be non-decreasing")
        last = float(value)


def _set_seed(seed: int, torch_mod) -> None:
    random.seed(seed)
    torch_mod.manual_seed(seed)
    if torch_mod.cuda.is_available():
        torch_mod.cuda.manual_seed_all(seed)


class PiThetaTorchModel:  # pragma: no cover - exercised by runtime scripts
    def __init__(self, cfg: TrainConfig):
        torch_mod, AutoModel, _ = _optional_imports()
        self.torch = torch_mod
        self.encoder = AutoModel.from_pretrained(cfg.base_model, local_files_only=False)
        hidden_size = int(getattr(self.encoder.config, "hidden_size"))
        self.wall_head = None
        self.ordinal_head = None
        if cfg.use_ordinal:
            self.ordinal_head = torch_mod.nn.Linear(hidden_size, 4 * 4)
        else:
            self.wall_head = torch_mod.nn.Linear(hidden_size, 4)
        self.polarity_head = torch_mod.nn.Linear(hidden_size, 4 * 3)
        self.cfg = cfg
        self._apply_lora_if_enabled()

    def _apply_lora_if_enabled(self) -> None:
        if not self.cfg.lora_enabled:
            return
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except Exception as exc:
            raise RuntimeError(f"lora requested but peft is unavailable: {exc}") from exc
        lora_cfg = LoraConfig(
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=self.cfg.lora_target_modules,
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
        )
        self.encoder = get_peft_model(self.encoder, lora_cfg)

    def parameters(self):
        out = list(self.encoder.parameters()) + list(self.polarity_head.parameters())
        if self.cfg.use_ordinal and self.ordinal_head is not None:
            out.extend(list(self.ordinal_head.parameters()))
        elif self.wall_head is not None:
            out.extend(list(self.wall_head.parameters()))
        return out

    def to(self, device):
        self.encoder.to(device)
        self.polarity_head.to(device)
        if self.ordinal_head is not None:
            self.ordinal_head.to(device)
        if self.wall_head is not None:
            self.wall_head.to(device)
        return self

    def train(self):
        self.encoder.train()
        self.polarity_head.train()
        if self.ordinal_head is not None:
            self.ordinal_head.train()
        if self.wall_head is not None:
            self.wall_head.train()

    def eval(self):
        self.encoder.eval()
        self.polarity_head.eval()
        if self.ordinal_head is not None:
            self.ordinal_head.eval()
        if self.wall_head is not None:
            self.wall_head.eval()

    def forward(self, tokens):
        encoded = self.encoder(**tokens)
        mask = tokens["attention_mask"].unsqueeze(-1).expand(encoded.last_hidden_state.size()).float()
        pooled = (encoded.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        if self.cfg.use_ordinal and self.ordinal_head is not None:
            primary_logits = self.ordinal_head(pooled).reshape((-1, 4, 4))
        elif self.wall_head is not None:
            primary_logits = self.wall_head(pooled)
        else:
            raise RuntimeError("invalid pitheta model head state")
        polarity_logits = self.polarity_head(pooled).reshape((-1, 4, 3))
        return primary_logits, polarity_logits


def _collate_batch(tokenizer, torch_mod, rows: List[Dict[str, Any]], max_len: int, device):
    texts = [str(r["text"]) for r in rows]
    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}
    y_wall = torch_mod.tensor([list(map(float, r["wall_labels"])) for r in rows], dtype=torch_mod.float32, device=device)
    y_pressure = torch_mod.tensor([list(map(int, r["pressure_level"])) for r in rows], dtype=torch_mod.long, device=device)
    y_pol = torch_mod.tensor([[int(x) + 1 for x in r["polarity"]] for r in rows], dtype=torch_mod.long, device=device)
    attack = torch_mod.tensor([int(r.get("is_attack", 0)) for r in rows], dtype=torch_mod.long, device=device)
    return tokens, y_wall, y_pressure, y_pol, attack


def _iter_batches(rows: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]


def _pick_rows(rows: List[Dict[str, Any]], *, count: int, seed: int) -> List[Dict[str, Any]]:
    if count <= 0 or not rows:
        return []
    if count >= len(rows):
        out = list(rows)
        random.Random(seed).shuffle(out)
        return out
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(rows)), k=int(count)))
    out = [rows[i] for i in idx]
    rng.shuffle(out)
    return out


def _resolve_calibration_rows(
    *,
    cfg: TrainConfig,
    data_dir: str,
    base_rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    mode = str(cfg.calibration_source_mode).lower()
    base = list(base_rows)
    meta: Dict[str, Any] = {
        "source_mode": mode,
        "dataset_rows": int(len(base)),
        "gold_rows": 0,
        "selected_dataset_rows": 0,
        "selected_gold_rows": 0,
    }
    if mode == "dataset":
        meta["selected_dataset_rows"] = int(len(base))
        return base, meta

    gold_path = Path(cfg.calibration_gold_slice_path)
    if not gold_path.is_absolute():
        gold_path = (Path(data_dir).resolve().parent / gold_path).resolve()
        if not gold_path.exists():
            gold_path = (Path.cwd() / cfg.calibration_gold_slice_path).resolve()
    if not gold_path.exists():
        raise ValueError(f"calibration gold_slice file not found: {cfg.calibration_gold_slice_path}")
    gold_rows = load_pitheta_jsonl(gold_path.as_posix())
    if not gold_rows:
        raise ValueError("calibration gold_slice file is empty")
    meta["gold_rows"] = int(len(gold_rows))
    meta["gold_slice_path"] = gold_path.as_posix()

    if mode == "gold_only":
        meta["selected_gold_rows"] = int(len(gold_rows))
        return list(gold_rows), meta

    target_total = max(len(base), len(gold_rows))
    target_gold = int(round(float(target_total) * float(cfg.calibration_gold_ratio)))
    target_weak = int(round(float(target_total) * float(cfg.calibration_weak_ratio)))
    if target_gold <= 0:
        target_gold = min(1, len(gold_rows))
    if target_weak <= 0 and base:
        target_weak = 1
    selected_gold = _pick_rows(gold_rows, count=min(target_gold, len(gold_rows)), seed=cfg.seed + 911)
    selected_weak = _pick_rows(base, count=min(target_weak, len(base)), seed=cfg.seed + 977)
    mixed = list(selected_gold) + list(selected_weak)
    random.Random(cfg.seed + 991).shuffle(mixed)
    if not mixed:
        raise ValueError("calibration blended mode selected zero rows")
    meta["selected_gold_rows"] = int(len(selected_gold))
    meta["selected_dataset_rows"] = int(len(selected_weak))
    meta["target_total"] = int(target_total)
    return mixed, meta


def _predict_attack_from_logits(*, model: PiThetaTorchModel, primary_logits, polarity_logits):
    torch_mod = model.torch
    if model.cfg.use_ordinal:
        ord_prob = torch_mod.softmax(primary_logits, dim=-1)
        pressure_map = torch_mod.tensor(model.cfg.pressure_map, dtype=torch_mod.float32, device=ord_prob.device)
        expected = (ord_prob * pressure_map.view(1, 1, 4)).sum(dim=-1)
        pol_pred = torch_mod.argmax(polarity_logits, dim=-1) - 1
        gated = torch_mod.where(pol_pred == 1, expected, torch_mod.zeros_like(expected))
        return (gated.max(dim=1).values >= float(model.cfg.pressure_map[1])).long()
    wall_prob = torch_mod.sigmoid(primary_logits)
    return (wall_prob.max(dim=1).values >= 0.5).long()


def _compute_dev_metrics(model: PiThetaTorchModel, tokenizer, rows: List[Dict[str, Any]], cfg: TrainConfig, device) -> Dict[str, float]:
    torch_mod = model.torch
    if not rows:
        return {
            "dev_attack_recall": 0.0,
            "dev_benign_fpr": 0.0,
        }
    model.eval()
    attack_tp = 0
    attack_total = 0
    benign_fp = 0
    benign_total = 0
    with torch_mod.no_grad():
        for batch_rows in _iter_batches(rows, cfg.batch_size):
            tokens, _, _, _, attack = _collate_batch(tokenizer, torch_mod, batch_rows, cfg.max_len, device)
            primary_logits, polarity_logits = model.forward(tokens)
            pred_attack = _predict_attack_from_logits(model=model, primary_logits=primary_logits, polarity_logits=polarity_logits)
            for i in range(pred_attack.shape[0]):
                is_attack = int(attack[i].item())
                pred = int(pred_attack[i].item())
                if is_attack == 1:
                    attack_total += 1
                    if pred == 1:
                        attack_tp += 1
                else:
                    benign_total += 1
                    if pred == 1:
                        benign_fp += 1
    recall = float(attack_tp) / float(attack_total) if attack_total > 0 else 0.0
    benign_fpr = float(benign_fp) / float(benign_total) if benign_total > 0 else 0.0
    return {
        "dev_attack_recall": recall,
        "dev_benign_fpr": benign_fpr,
    }


def _collect_logits_for_calibration(
    model: PiThetaTorchModel,
    tokenizer,
    rows: List[Dict[str, Any]],
    cfg: TrainConfig,
    device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    torch_mod = model.torch
    ord_logits_all: List[np.ndarray] = []
    ord_targets_all: List[np.ndarray] = []
    pol_logits_all: List[np.ndarray] = []
    pol_targets_all: List[np.ndarray] = []
    model.eval()
    with torch_mod.no_grad():
        for batch_rows in _iter_batches(rows, cfg.batch_size):
            tokens, _, y_pressure, y_pol, _ = _collate_batch(tokenizer, torch_mod, batch_rows, cfg.max_len, device)
            primary_logits, polarity_logits = model.forward(tokens)
            if cfg.use_ordinal:
                ord_logits_all.append(primary_logits.detach().cpu().numpy().astype(np.float32))
            else:
                wall_logits = primary_logits.detach().cpu().numpy().astype(np.float32)
                # Legacy: convert binary wall logits to pseudo-ordinal logits for compatibility in calibration payload.
                pseudo = np.zeros((wall_logits.shape[0], 4, 4), dtype=np.float32)
                pseudo[:, :, 0] = -np.abs(wall_logits)
                pseudo[:, :, 1] = wall_logits * 0.5
                pseudo[:, :, 2] = wall_logits
                pseudo[:, :, 3] = wall_logits * 1.5
                ord_logits_all.append(pseudo)
            ord_targets_all.append(y_pressure.detach().cpu().numpy().astype(np.int64))
            pol_logits_all.append(polarity_logits.detach().cpu().numpy().astype(np.float32))
            pol_targets_all.append(y_pol.detach().cpu().numpy().astype(np.int64))
    if not ord_logits_all:
        return (
            np.zeros((0, 4, 4), dtype=np.float32),
            np.zeros((0, 4), dtype=np.int64),
            np.zeros((0, 4, 3), dtype=np.float32),
            np.zeros((0, 4), dtype=np.int64),
        )
    return (
        np.vstack(ord_logits_all),
        np.vstack(ord_targets_all),
        np.vstack(pol_logits_all),
        np.vstack(pol_targets_all),
    )


def train_pitheta_lora(  # pragma: no cover - exercised by runtime scripts
    *,
    train_config_path: str,
    data_dir: str,
    output_dir: str,
    resume_from: str | None = None,
) -> Dict[str, Any]:
    cfg = load_train_config(train_config_path)
    torch_mod, _, _ = _optional_imports()

    train_rows = load_pitheta_jsonl(str(Path(data_dir) / "train.jsonl"))
    dev_rows = load_pitheta_jsonl(str(Path(data_dir) / "dev.jsonl"))
    holdout_rows = load_pitheta_jsonl(str(Path(data_dir) / "holdout.jsonl")) if (Path(data_dir) / "holdout.jsonl").exists() else []
    calibration_rows = (
        load_pitheta_jsonl(str(Path(data_dir) / "calibration.jsonl"))
        if (Path(data_dir) / "calibration.jsonl").exists()
        else []
    )
    if not train_rows:
        raise ValueError("train.jsonl is empty")
    if cfg.fit_temperature and cfg.temperature_split == "calibration" and not calibration_rows:
        raise ValueError("temperature_split=calibration requires non-empty calibration.jsonl in data_dir")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _set_seed(cfg.seed, torch_mod)
    device = torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")

    model = PiThetaTorchModel(cfg)
    model.to(device)
    tokenizer = _load_tokenizer(cfg.base_model, local_files_only=False)
    optimizer = torch_mod.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    bce = torch_mod.nn.BCEWithLogitsLoss()

    ordinal_ce_per_wall = [
        torch_mod.nn.CrossEntropyLoss(
            weight=torch_mod.tensor(cfg.ordinal_class_weights[w], dtype=torch_mod.float32, device=device),
            reduction="mean",
        )
        for w in range(4)
    ]
    polarity_ce_per_wall = [
        torch_mod.nn.CrossEntropyLoss(
            weight=torch_mod.tensor(cfg.polarity_class_weights[w], dtype=torch_mod.float32, device=device),
            reduction="mean",
        )
        for w in range(4)
    ]

    best: Dict[str, Any] = {"metric": -1.0, "epoch": 0}
    best_ckpt = out / "best"
    best_ckpt.mkdir(parents=True, exist_ok=True)
    patience_left = cfg.early_stopping_patience
    metrics_path = out / "train_metrics.jsonl"
    if resume_from:
        _ = str(resume_from)  # reserved for future resume support

    rows_shuffled = list(train_rows)
    for epoch in range(1, cfg.epochs + 1):
        random.Random(cfg.seed + epoch).shuffle(rows_shuffled)
        model.train()
        running_loss = 0.0
        step_count = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_rows in _iter_batches(rows_shuffled, cfg.batch_size):
            tokens, y_wall, y_pressure, y_pol, _ = _collate_batch(tokenizer, torch_mod, batch_rows, cfg.max_len, device)
            primary_logits, polarity_logits = model.forward(tokens)

            if cfg.use_ordinal:
                ord_losses = []
                for wall_idx in range(4):
                    ord_losses.append(ordinal_ce_per_wall[wall_idx](primary_logits[:, wall_idx, :], y_pressure[:, wall_idx]))
                ord_loss = torch_mod.stack(ord_losses).mean()
                pol_losses = []
                for wall_idx in range(4):
                    pol_losses.append(polarity_ce_per_wall[wall_idx](polarity_logits[:, wall_idx, :], y_pol[:, wall_idx]))
                pol_loss = torch_mod.stack(pol_losses).mean()
                loss = (cfg.loss_weight_ordinal * ord_loss) + (cfg.loss_weight_polarity * pol_loss)
            else:
                wall_loss = bce(primary_logits, y_wall)
                pol_losses = []
                for wall_idx in range(4):
                    logits_wall = polarity_logits[:, wall_idx, :]
                    target_wall = y_pol[:, wall_idx]
                    mask = y_wall[:, wall_idx] > 0.5
                    if bool(mask.any().item()):
                        pol_losses.append(polarity_ce_per_wall[wall_idx](logits_wall[mask], target_wall[mask]))
                pol_loss = torch_mod.stack(pol_losses).mean() if pol_losses else torch_mod.tensor(0.0, device=device)
                loss = wall_loss + (cfg.loss_weight_polarity * pol_loss)

            loss.backward()
            step_count += 1
            if step_count % cfg.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss.detach().cpu().item())

        dev_metrics = _compute_dev_metrics(model, tokenizer, dev_rows, cfg, device)
        epoch_report = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, step_count),
            **dev_metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(epoch_report, ensure_ascii=True) + "\n")

        candidate_metric = float(dev_metrics.get(cfg.early_stopping_metric, 0.0))
        benign_fpr = float(dev_metrics.get("dev_benign_fpr", 1.0))
        improved = candidate_metric > float(best["metric"]) and benign_fpr <= cfg.benign_fpr_ceiling
        if improved:
            best = {"metric": candidate_metric, "epoch": epoch, **dev_metrics}
            patience_left = cfg.early_stopping_patience
            heads_payload = {
                "head_version": "ordinal_v2" if cfg.use_ordinal else "legacy_v1",
                "polarity_head_state_dict": model.polarity_head.state_dict(),
            }
            if cfg.use_ordinal and model.ordinal_head is not None:
                heads_payload["ordinal_head_state_dict"] = model.ordinal_head.state_dict()
            elif model.wall_head is not None:
                heads_payload["wall_head_state_dict"] = model.wall_head.state_dict()
            torch_mod.save(heads_payload, str(best_ckpt / "heads.pt"))
            tokenizer.save_pretrained(str(best_ckpt / "tokenizer"))
            if cfg.lora_enabled and hasattr(model.encoder, "save_pretrained"):
                model.encoder.save_pretrained(str(best_ckpt / "adapter"))
            elif hasattr(model.encoder, "state_dict"):
                torch_mod.save(model.encoder.state_dict(), str(best_ckpt / "encoder_state.pt"))

            if cfg.fit_temperature:
                if cfg.temperature_split == "dev":
                    temp_rows = dev_rows
                elif cfg.temperature_split == "holdout":
                    temp_rows = holdout_rows
                else:
                    temp_rows = calibration_rows
                temp_rows, temp_meta = _resolve_calibration_rows(
                    cfg=cfg,
                    data_dir=data_dir,
                    base_rows=temp_rows,
                )
                ord_logits, ord_targets, pol_logits, pol_targets = _collect_logits_for_calibration(
                    model=model,
                    tokenizer=tokenizer,
                    rows=temp_rows,
                    cfg=cfg,
                    device=device,
                )
                payload = build_temperature_payload(
                    ordinal_logits=ord_logits,
                    ordinal_targets=ord_targets,
                    polarity_logits=pol_logits,
                    polarity_targets=pol_targets,
                )
                payload.update(
                    {
                        "head_version": heads_payload["head_version"],
                        "temperature_split": cfg.temperature_split,
                        "samples": int(ord_logits.shape[0]),
                        "calibration_source": temp_meta,
                    }
                )
                out_path = Path(cfg.temperature_output)
                if not out_path.is_absolute():
                    out_path = out / out_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        else:
            patience_left -= 1
            if patience_left < 0:
                break

    manifest = {
        "base_model": cfg.base_model,
        "task": cfg.task,
        "head_version": "ordinal_v2" if cfg.use_ordinal else "legacy_v1",
        "walls": ["override_instructions", "secret_exfiltration", "tool_or_action_abuse", "policy_evasion"],
        "checkpoint_dir": best_ckpt.as_posix(),
        "best": best,
        "train_config": {
            "max_len": cfg.max_len,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "epochs": cfg.epochs,
            "fit_temperature": cfg.fit_temperature,
            "temperature_split": cfg.temperature_split,
            "temperature_output": cfg.temperature_output,
            "calibration_source_mode": cfg.calibration_source_mode,
            "calibration_gold_slice_path": cfg.calibration_gold_slice_path,
            "calibration_gold_ratio": cfg.calibration_gold_ratio,
            "calibration_weak_ratio": cfg.calibration_weak_ratio,
            "lora_enabled": cfg.lora_enabled,
            "lora": {
                "r": cfg.lora_r,
                "alpha": cfg.lora_alpha,
                "dropout": cfg.lora_dropout,
                "target_modules": list(cfg.lora_target_modules),
            },
        },
    }
    (best_ckpt / "model_manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    (out / "model_manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    return manifest
