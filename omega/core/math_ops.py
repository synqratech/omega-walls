"""Math primitives for Omega core with strict finite-value contracts."""

from __future__ import annotations

import math

import numpy as np


def _finite_1d(value: np.ndarray, *, name: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric vector") from exc
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} must contain only finite values")
    return arr.astype(float, copy=True)


def floor_epsilon(v: np.ndarray, epsilon: float) -> np.ndarray:
    out = _finite_1d(v, name="v")
    epsilon_value = float(epsilon)
    if not math.isfinite(epsilon_value) or epsilon_value < 0.0:
        raise ValueError("epsilon must be finite and >= 0")
    if bool(np.any(out < 0.0)):
        raise ValueError("v must be nonnegative")
    out[out < epsilon_value] = 0.0
    return out


def phi_alpha(v: np.ndarray, alpha: float) -> np.ndarray:
    arr = _finite_1d(v, name="v")
    alpha_value = float(alpha)
    if not math.isfinite(alpha_value) or alpha_value <= 0.0:
        raise ValueError("alpha must be finite and > 0")
    if bool(np.any(arr < 0.0)):
        raise ValueError("v must be nonnegative")
    out = 1.0 - np.exp(-alpha_value * arr)
    if not bool(np.all(np.isfinite(out))):
        raise ValueError("phi_alpha produced non-finite output")
    return out


def synergy_operator(p: np.ndarray, S: np.ndarray) -> np.ndarray:
    p_arr = _finite_1d(p, name="p")
    try:
        s_arr = np.asarray(S, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("S must be a numeric matrix") from exc
    k_count = len(p_arr)
    if s_arr.shape != (k_count, k_count):
        raise ValueError(f"S must have shape ({k_count}, {k_count})")
    if not bool(np.all(np.isfinite(s_arr))) or bool(np.any(s_arr < 0.0)):
        raise ValueError("S must contain only finite nonnegative values")
    if bool(np.any(p_arr < 0.0)) or bool(np.any(p_arr > 1.0)):
        raise ValueError("p must be in [0,1]")
    out = np.zeros(k_count, dtype=float)
    for k in range(k_count):
        val = 0.0
        for ell in range(k_count):
            if k == ell:
                continue
            val += s_arr[k, ell] * min(p_arr[k], p_arr[ell])
        out[k] = val
    if not bool(np.all(np.isfinite(out))):
        raise ValueError("synergy operator produced non-finite output")
    return out


def clip01(v: np.ndarray) -> np.ndarray:
    arr = _finite_1d(v, name="v")
    return np.clip(arr, 0.0, 1.0)
