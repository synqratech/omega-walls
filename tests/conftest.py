from __future__ import annotations

import numpy as np
import pytest

from omega.config.loader import load_resolved_config
from omega.core.omega_core import OmegaCoreV1
from omega.core.params import omega_params_from_config
from omega.interfaces.contracts_v1 import OmegaState
from omega.policy.off_policy_v1 import OffPolicyV1
from omega.projector.pi0_intent_v2 import Pi0IntentAwareV2
from omega.tools.tool_gateway import ToolGatewayV1


@pytest.fixture(scope="session")
def resolved_config() -> dict:
    return load_resolved_config(profile="dev").resolved


@pytest.fixture(scope="session")
def projector(resolved_config: dict) -> Pi0IntentAwareV2:
    return Pi0IntentAwareV2(resolved_config)


@pytest.fixture
def omega_core(resolved_config: dict) -> OmegaCoreV1:
    return OmegaCoreV1(omega_params_from_config(resolved_config))


@pytest.fixture
def omega_state() -> OmegaState:
    return OmegaState(session_id="sess-test", m=np.zeros(4, dtype=float), step=0)


@pytest.fixture
def off_policy(resolved_config: dict) -> OffPolicyV1:
    return OffPolicyV1(resolved_config)


@pytest.fixture
def gateway(resolved_config: dict) -> ToolGatewayV1:
    return ToolGatewayV1(resolved_config)
