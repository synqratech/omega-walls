"""Math primitives for omega core."""

from __future__ import annotations

import numpy as np


def floor_epsilon(v: np.ndarray, epsilon: float) -> np.ndarray:
    out = np.array(v, dtype=float).copy()
    out[out < epsilon] = 0.0
    return out


def phi_alpha(v: np.ndarray, alpha: float) -> np.ndarray:
    return 1.0 - np.exp(-alpha * np.array(v, dtype=float))


def synergy_operator(p: np.ndarray, S: np.ndarray) -> np.ndarray:
    p = np.array(p, dtype=float)
    K = len(p)
    out = np.zeros(K, dtype=float)
    for k in range(K):
        val = 0.0
        for ell in range(K):
            if k == ell:
                continue
            val += S[k, ell] * min(p[k], p[ell])
        out[k] = val
    return out


def clip01(v: np.ndarray) -> np.ndarray:
    return np.clip(np.array(v, dtype=float), 0.0, 1.0)
