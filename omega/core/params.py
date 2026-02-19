"""Parameter helpers."""

from __future__ import annotations

import numpy as np

from omega.interfaces.contracts_v1 import OmegaParams


def omega_params_from_config(config: dict) -> OmegaParams:
    ocfg = config["omega"]
    off = ocfg["off"]
    return OmegaParams(
        walls=list(ocfg["walls"]),
        epsilon=float(ocfg["epsilon"]),
        alpha=float(ocfg["alpha"]),
        beta=float(ocfg["beta"]),
        lam=float(ocfg["lambda"]),
        S=np.array(ocfg["S"], dtype=float),
        off_tau=float(off["tau"]),
        off_Theta=float(off["Theta"]),
        off_Sigma=float(off["Sigma"]),
        off_theta=float(off["theta"]),
        off_N=int(off["N"]),
        attrib_gamma=float(ocfg["attribution"]["gamma"]),
    )
