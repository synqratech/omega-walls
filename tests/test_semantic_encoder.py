from __future__ import annotations

import numpy as np
import pytest

from omega.projector.semantic_encoder import cosine_similarity_matrix, mean_pool_hidden


def test_cosine_similarity_matrix_invariants():
    a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    b = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    sims = cosine_similarity_matrix(a, b)

    assert sims.shape == (2, 2)
    assert pytest.approx(sims[0, 0], rel=1e-6) == 1.0
    assert sims[1, 0] < 1e-6
    assert 0.70 < sims[0, 1] < 0.71
    assert 0.70 < sims[1, 1] < 0.71


def test_mean_pool_hidden_masks_padding():
    torch = pytest.importorskip("torch")
    hidden = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [999.0, 999.0]],
            [[2.0, 2.0], [4.0, 4.0], [6.0, 6.0]],
        ],
        dtype=torch.float32,
    )
    attn = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long)
    pooled = mean_pool_hidden(hidden, attn)

    assert tuple(pooled.shape) == (2, 2)
    assert pooled[0, 0].item() == pytest.approx(2.0, rel=1e-6)
    assert pooled[0, 1].item() == pytest.approx(3.0, rel=1e-6)
    assert pooled[1, 0].item() == pytest.approx(4.0, rel=1e-6)
    assert pooled[1, 1].item() == pytest.approx(4.0, rel=1e-6)
