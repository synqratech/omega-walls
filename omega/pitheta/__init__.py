"""PiTheta training and evaluation helpers.

This module intentionally avoids eager imports of dataset builder internals,
because those pull optional dev-only dependencies (for example, ``redteam``).
"""

from typing import Any

__all__ = ["PI_THETA_SCHEMA_VERSION", "WALL_ORDER", "build_pitheta_dataset_artifacts", "evaluate_pitheta_gates"]


def __getattr__(name: str) -> Any:
    if name in {"PI_THETA_SCHEMA_VERSION", "WALL_ORDER", "build_pitheta_dataset_artifacts"}:
        from omega.pitheta import dataset_builder as _dataset_builder

        return getattr(_dataset_builder, name)
    if name == "evaluate_pitheta_gates":
        from omega.pitheta.eval_gates import evaluate_pitheta_gates

        return evaluate_pitheta_gates
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
