from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple


def override_score(self: Any, t: str, t_ns: str, tokens: Sequence[str], struct_count: int) -> Tuple[float, Dict[str, Any], int]:
    return self._override_score_legacy(t, t_ns, tokens, struct_count)
