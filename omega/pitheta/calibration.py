"""Temperature scaling utilities for PiTheta heads."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class TemperatureFitResult:
    temperatures: List[float]
    nll_before: float
    nll_after: float


def _nll_numpy(logits: np.ndarray, targets: np.ndarray) -> float:
    if logits.size == 0 or targets.size == 0:
        return 0.0
    logits = logits.astype(np.float64, copy=False)
    targets = targets.astype(np.int64, copy=False)
    max_per_row = np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits - max_per_row)
    probs = exp / np.clip(np.sum(exp, axis=1, keepdims=True), 1e-12, None)
    rows = np.arange(targets.shape[0], dtype=np.int64)
    picked = np.clip(probs[rows, targets], 1e-12, 1.0)
    return float(-np.mean(np.log(picked)))


def apply_temperature(logits: np.ndarray, temperatures: List[float]) -> np.ndarray:
    if logits.ndim != 3:
        raise ValueError("logits must be shape (N, K, C)")
    if len(temperatures) != logits.shape[1]:
        raise ValueError("temperatures length must match K")
    out = logits.astype(np.float32, copy=True)
    for wall_idx, temp in enumerate(temperatures):
        t = max(float(temp), 1e-6)
        out[:, wall_idx, :] = out[:, wall_idx, :] / t
    return out


def fit_temperature_per_wall(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    min_temp: float = 0.05,
    max_temp: float = 10.0,
) -> TemperatureFitResult:
    """Fit one scalar temperature per wall using lightweight grid search."""

    if logits.ndim != 3:
        raise ValueError("logits must be shape (N, K, C)")
    if targets.ndim != 2:
        raise ValueError("targets must be shape (N, K)")
    if logits.shape[0] != targets.shape[0] or logits.shape[1] != targets.shape[1]:
        raise ValueError("targets shape must match logits first two dimensions")
    n, k, _ = logits.shape
    if n == 0:
        return TemperatureFitResult(temperatures=[1.0] * k, nll_before=0.0, nll_after=0.0)

    grid = [0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
    grid = [float(x) for x in grid if float(min_temp) <= float(x) <= float(max_temp)]
    if 1.0 not in grid:
        grid.append(1.0)
        grid = sorted(set(grid))

    best_t: List[float] = []
    for wall_idx in range(k):
        wall_logits = logits[:, wall_idx, :]
        wall_targets = targets[:, wall_idx]
        best_loss = math.inf
        best_temp = 1.0
        for temp in grid:
            loss = _nll_numpy(wall_logits / temp, wall_targets)
            if loss < best_loss:
                best_loss = loss
                best_temp = float(temp)
        best_t.append(best_temp)

    before = _nll_numpy(logits.reshape((-1, logits.shape[-1])), targets.reshape((-1,)))
    after_logits = apply_temperature(logits, best_t)
    after = _nll_numpy(after_logits.reshape((-1, after_logits.shape[-1])), targets.reshape((-1,)))
    return TemperatureFitResult(temperatures=best_t, nll_before=before, nll_after=after)


def build_temperature_payload(
    *,
    ordinal_logits: np.ndarray,
    ordinal_targets: np.ndarray,
    polarity_logits: np.ndarray,
    polarity_targets: np.ndarray,
) -> Dict[str, object]:
    ord_fit = fit_temperature_per_wall(ordinal_logits, ordinal_targets)
    pol_fit = fit_temperature_per_wall(polarity_logits, polarity_targets)
    return {
        "schema_version": "pitheta_temperature_v1",
        "ordinal": {
            "temperatures": [float(x) for x in ord_fit.temperatures],
            "nll_before": float(ord_fit.nll_before),
            "nll_after": float(ord_fit.nll_after),
        },
        "polarity": {
            "temperatures": [float(x) for x in pol_fit.temperatures],
            "nll_before": float(pol_fit.nll_before),
            "nll_after": float(pol_fit.nll_after),
        },
    }

