"""Omega core implementation for v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from omega.core.math_ops import clip01, floor_epsilon, phi_alpha, synergy_operator
from omega.interfaces.contracts_v1 import (
    ContentItem,
    DocContribution,
    OmegaOffReasons,
    OmegaParams,
    OmegaState,
    OmegaStepResult,
    ProjectionResult,
)


@dataclass
class OmegaCoreV1:
    params: OmegaParams

    def step(
        self,
        state: OmegaState,
        items: List[ContentItem],
        projections: List[ProjectionResult],
    ) -> OmegaStepResult:
        if len(items) != len(projections):
            raise ValueError("items and projections must align 1:1")

        item_by_id = {it.doc_id: it for it in items}
        for proj in projections:
            if proj.doc_id not in item_by_id:
                raise ValueError(f"projection doc_id not found in items: {proj.doc_id}")

        v_docs = [floor_epsilon(proj.v, self.params.epsilon) for proj in projections]
        v_total = np.sum(v_docs, axis=0) if v_docs else np.zeros(len(self.params.walls), dtype=float)

        p0 = phi_alpha(v_total, self.params.alpha)
        p = clip01(p0 + self.params.beta * synergy_operator(p0, self.params.S))

        e_total = v_total * p
        m_prev = state.m.copy()
        m_next = self.params.lam * state.m + e_total

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
        for proj, v_doc in zip(projections, v_docs):
            item = item_by_id[proj.doc_id]
            e_doc = v_doc * p
            c = float(np.sum(e_doc))
            contribs.append(
                DocContribution(
                    doc_id=proj.doc_id,
                    source_id=item.source_id,
                    v=v_doc,
                    e=e_doc,
                    c=c,
                    evidence=proj.evidence,
                )
            )

        top_docs: List[str] = []
        if contribs:
            c_max = max(c.c for c in contribs)
            threshold = self.params.attrib_gamma * c_max
            top_docs = [c.doc_id for c in contribs if c.c >= threshold]

        state.m = m_next
        state.step += 1

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
        )
