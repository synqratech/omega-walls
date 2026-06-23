"""Omega core implementation for v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from omega.core.math_ops import clip01, floor_epsilon, phi_alpha, synergy_operator
from omega.interfaces.contracts_v1 import (
    ContentItem,
    DocContribution,
    K_V1,
    OmegaOffReasons,
    OmegaParams,
    OmegaState,
    OmegaStepResult,
    ProjectionResult,
    WALLS_V1,
)
from omega.validation.numeric import (
    finite_vector,
    validate_omega_params_values,
    validate_projection_values,
    validate_state_values,
)


@dataclass
class OmegaCoreV1:
    params: OmegaParams

    def __post_init__(self) -> None:
        self._validate_params()

    def _validate_params(self) -> None:
        # Re-run on every step as OmegaParams is mutable and may be changed by callers.
        values = validate_omega_params_values(
            walls=self.params.walls,
            expected_walls=WALLS_V1,
            epsilon=self.params.epsilon,
            alpha=self.params.alpha,
            beta=self.params.beta,
            lam=self.params.lam,
            synergy=self.params.S,
            off_tau=self.params.off_tau,
            off_theta_hard=self.params.off_Theta,
            off_sigma=self.params.off_Sigma,
            off_theta_multi=self.params.off_theta,
            off_n=self.params.off_N,
            attrib_gamma=self.params.attrib_gamma,
        )
        for key, value in values.items():
            setattr(self.params, key, value)

    @staticmethod
    def _validate_alignment(items: List[ContentItem], projections: List[ProjectionResult]) -> dict[str, ContentItem]:
        if len(items) != len(projections):
            raise ValueError("items and projections must align 1:1")
        item_ids = [str(item.doc_id) for item in items]
        projection_ids = [str(proj.doc_id) for proj in projections]
        if any(not value.strip() for value in item_ids):
            raise ValueError("item doc_id must be non-empty")
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("item doc_id values must be unique within a packet")
        if len(set(projection_ids)) != len(projection_ids):
            raise ValueError("projection doc_id values must be unique within a packet")
        if item_ids != projection_ids:
            missing = sorted(set(item_ids) - set(projection_ids))
            extra = sorted(set(projection_ids) - set(item_ids))
            raise ValueError(
                "items/projections must have identical doc_id order: "
                f"items={item_ids}, projections={projection_ids}, missing={missing}, extra={extra}"
            )
        return {item.doc_id: item for item in items}

    def step(
        self,
        state: OmegaState,
        items: List[ContentItem],
        projections: List[ProjectionResult],
    ) -> OmegaStepResult:
        self._validate_params()
        if not isinstance(state, OmegaState):
            raise ValueError("state must be OmegaState")
        state_m = validate_state_values(vector=state.m, wall_count=K_V1)
        if isinstance(state.step, bool) or int(state.step) != state.step or int(state.step) < 0:
            raise ValueError("state.step must be a nonnegative integer")

        item_by_id = self._validate_alignment(items, projections)
        v_docs: list[np.ndarray] = []
        for index, projection in enumerate(projections):
            if not isinstance(projection, ProjectionResult):
                raise ValueError(f"projections[{index}] must be ProjectionResult")
            vector, polarity, raw = validate_projection_values(
                vector=projection.v,
                polarity=projection.evidence.polarity,
                debug_scores_raw=projection.evidence.debug_scores_raw,
                wall_count=K_V1,
                name=f"projections[{index}]",
            )
            projection.v = vector
            projection.evidence.polarity = polarity
            projection.evidence.debug_scores_raw = raw
            v_docs.append(floor_epsilon(vector, self.params.epsilon))

        v_total = np.sum(v_docs, axis=0, dtype=float) if v_docs else np.zeros(K_V1, dtype=float)
        v_total = finite_vector(v_total, name="omega.v_total", length=K_V1, nonnegative=True)

        p0 = phi_alpha(v_total, self.params.alpha)
        p = clip01(p0 + self.params.beta * synergy_operator(p0, self.params.S))
        p = finite_vector(p, name="omega.p", length=K_V1, nonnegative=True)
        if bool(np.any(p > 1.0)):
            raise ValueError("omega.p must be in [0,1]")

        e_total = finite_vector(v_total * p, name="omega.e_total", length=K_V1, nonnegative=True)
        m_prev = state_m.copy()
        m_next = finite_vector(
            self.params.lam * state_m + e_total,
            name="omega.m_next",
            length=K_V1,
            nonnegative=True,
        )

        reason_spike = bool(float(np.max(p)) >= self.params.off_tau)
        reason_wall = bool(float(np.max(m_next)) >= self.params.off_Theta)
        reason_sum = bool(float(np.sum(m_next)) >= self.params.off_Sigma)
        reason_multi = bool(int(np.count_nonzero(m_next >= self.params.off_theta)) >= self.params.off_N)
        off = reason_spike or reason_wall or reason_sum or reason_multi

        reasons = OmegaOffReasons(
            reason_spike=reason_spike,
            reason_wall=reason_wall,
            reason_sum=reason_sum,
            reason_multi=reason_multi,
        )

        contribs: List[DocContribution] = []
        for projection, v_doc in zip(projections, v_docs):
            item = item_by_id[projection.doc_id]
            e_doc = finite_vector(v_doc * p, name=f"contrib[{projection.doc_id}].e", length=K_V1, nonnegative=True)
            c = float(np.sum(e_doc))
            if not np.isfinite(c) or c < 0.0:
                raise ValueError(f"contrib[{projection.doc_id}].c must be finite and nonnegative")
            contribs.append(
                DocContribution(
                    doc_id=projection.doc_id,
                    source_id=item.source_id,
                    v=v_doc,
                    e=e_doc,
                    c=c,
                    evidence=projection.evidence,
                )
            )

        top_docs: List[str] = []
        attribution_mode = "none"
        if contribs:
            c_max = max(contribution.c for contribution in contribs)
            # Historical state can trigger Off with zero current contribution. Never blame
            # current benign documents in that case.
            if c_max > 0.0:
                threshold = self.params.attrib_gamma * c_max
                top_docs = [contribution.doc_id for contribution in contribs if contribution.c >= threshold]
                attribution_mode = "current_packet"
            elif off:
                attribution_mode = "state_only"

        # Commit state only after all validation and calculations complete.
        state.m = m_next.copy()
        state.step = int(state.step) + 1

        return OmegaStepResult(
            session_id=state.session_id,
            step=state.step,
            v_total=v_total,
            p=p,
            m_prev=m_prev,
            m_next=m_next,
            off=off,
            reasons=reasons,
            top_docs=top_docs,
            contribs=contribs,
            attribution_mode=attribution_mode,
        )
