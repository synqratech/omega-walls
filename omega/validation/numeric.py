"""Strict numeric validation for all Omega wall-space boundaries.

The safety core must never accept NaN/Inf, ragged arrays, wrong wall counts,
or negative pressure/state values.  Python/NumPy comparisons with NaN are
fail-open, so validation is deliberately duplicated at construction,
persistence, and execution boundaries.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np


class NumericContractError(ValueError):
    """Raised when a numeric safety contract is violated."""


def finite_float(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_exclusive: bool = False,
    maximum_exclusive: bool = False,
) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NumericContractError(f"{name} must be a finite number") from exc
    if not math.isfinite(out):
        raise NumericContractError(f"{name} must be finite")
    if minimum is not None:
        if minimum_exclusive and not out > minimum:
            raise NumericContractError(f"{name} must be > {minimum}")
        if not minimum_exclusive and out < minimum:
            raise NumericContractError(f"{name} must be >= {minimum}")
    if maximum is not None:
        if maximum_exclusive and not out < maximum:
            raise NumericContractError(f"{name} must be < {maximum}")
        if not maximum_exclusive and out > maximum:
            raise NumericContractError(f"{name} must be <= {maximum}")
    return out


def finite_vector(
    value: Any,
    *,
    name: str,
    length: int,
    nonnegative: bool = False,
) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NumericContractError(f"{name} must be a numeric vector of length {length}") from exc
    if arr.ndim != 1 or arr.shape != (int(length),):
        raise NumericContractError(f"{name} must have shape ({int(length)},), got {arr.shape}")
    if not bool(np.all(np.isfinite(arr))):
        raise NumericContractError(f"{name} must contain only finite values")
    if nonnegative and bool(np.any(arr < 0.0)):
        raise NumericContractError(f"{name} must contain only nonnegative values")
    return arr.astype(float, copy=True)


def finite_matrix(
    value: Any,
    *,
    name: str,
    shape: tuple[int, int],
    nonnegative: bool = False,
) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NumericContractError(f"{name} must be a numeric matrix with shape {shape}") from exc
    if arr.ndim != 2 or arr.shape != shape:
        raise NumericContractError(f"{name} must have shape {shape}, got {arr.shape}")
    if not bool(np.all(np.isfinite(arr))):
        raise NumericContractError(f"{name} must contain only finite values")
    if nonnegative and bool(np.any(arr < 0.0)):
        raise NumericContractError(f"{name} must contain only nonnegative values")
    return arr.astype(float, copy=True)


def finite_sequence(values: Iterable[Any], *, name: str, length: int) -> list[float]:
    seq = list(values)
    if len(seq) != int(length):
        raise NumericContractError(f"{name} must contain exactly {int(length)} values")
    return [finite_float(value, name=f"{name}[{idx}]") for idx, value in enumerate(seq)]


def validate_projection_values(
    *,
    vector: Any,
    polarity: Sequence[Any],
    debug_scores_raw: Sequence[Any],
    wall_count: int,
    name: str = "projection",
) -> tuple[np.ndarray, list[int], list[float]]:
    arr = finite_vector(vector, name=f"{name}.v", length=wall_count, nonnegative=True)
    pol = list(polarity)
    if len(pol) != int(wall_count):
        raise NumericContractError(f"{name}.evidence.polarity must contain exactly {wall_count} values")
    normalized_pol: list[int] = []
    for idx, value in enumerate(pol):
        if isinstance(value, bool):
            raise NumericContractError(f"{name}.evidence.polarity[{idx}] must be -1, 0, or 1")
        try:
            ivalue = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise NumericContractError(f"{name}.evidence.polarity[{idx}] must be -1, 0, or 1") from exc
        if ivalue not in {-1, 0, 1} or float(value) != float(ivalue):
            raise NumericContractError(f"{name}.evidence.polarity[{idx}] must be -1, 0, or 1")
        normalized_pol.append(ivalue)
    raw = finite_sequence(debug_scores_raw, name=f"{name}.evidence.debug_scores_raw", length=wall_count)
    return arr, normalized_pol, raw


def validate_state_values(*, vector: Any, wall_count: int, name: str = "state.m") -> np.ndarray:
    return finite_vector(vector, name=name, length=wall_count, nonnegative=True)


def validate_omega_params_values(
    *,
    walls: Sequence[str],
    expected_walls: Sequence[str],
    epsilon: Any,
    alpha: Any,
    beta: Any,
    lam: Any,
    synergy: Any,
    off_tau: Any,
    off_theta_hard: Any,
    off_sigma: Any,
    off_theta_multi: Any,
    off_n: Any,
    attrib_gamma: Any,
) -> dict[str, Any]:
    normalized_walls = [str(x) for x in walls]
    if normalized_walls != [str(x) for x in expected_walls]:
        raise NumericContractError("omega.walls must exactly match the v1 wall order")
    k = len(normalized_walls)
    matrix = finite_matrix(synergy, name="omega.S", shape=(k, k), nonnegative=True)
    if not bool(np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=0.0)):
        raise NumericContractError("omega.S diagonal must be exactly zero")
    try:
        n_value = int(off_n)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NumericContractError("omega.off.N must be an integer") from exc
    if isinstance(off_n, bool) or float(off_n) != float(n_value) or n_value < 1 or n_value > k:
        raise NumericContractError(f"omega.off.N must be an integer in 1..{k}")
    return {
        "walls": normalized_walls,
        "epsilon": finite_float(epsilon, name="omega.epsilon", minimum=0.0),
        "alpha": finite_float(alpha, name="omega.alpha", minimum=0.0, minimum_exclusive=True),
        "beta": finite_float(beta, name="omega.beta", minimum=0.0),
        "lam": finite_float(lam, name="omega.lambda", minimum=0.0, maximum=1.0, minimum_exclusive=True, maximum_exclusive=True),
        "S": matrix,
        "off_tau": finite_float(off_tau, name="omega.off.tau", minimum=0.0, maximum=1.0, minimum_exclusive=True),
        "off_Theta": finite_float(off_theta_hard, name="omega.off.Theta", minimum=0.0, minimum_exclusive=True),
        "off_Sigma": finite_float(off_sigma, name="omega.off.Sigma", minimum=0.0, minimum_exclusive=True),
        "off_theta": finite_float(off_theta_multi, name="omega.off.theta", minimum=0.0, minimum_exclusive=True),
        "off_N": n_value,
        "attrib_gamma": finite_float(attrib_gamma, name="omega.attribution.gamma", minimum=0.0, maximum=1.0, minimum_exclusive=True),
    }
