from __future__ import annotations

from typing import Any

from omega.interfaces.contracts_v1 import ContentItem, ProjectionResult


def project_item(self: Any, item: ContentItem) -> ProjectionResult:
    return self._project_legacy(item)
