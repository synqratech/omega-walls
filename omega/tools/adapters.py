"""Tool adapter registry and default adapters for local execution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

from omega.interfaces.contracts_v1 import ToolRequest
from omega.security.paths import resolve_contained_path


ToolAdapterFunc = Callable[[ToolRequest, Dict[str, Any]], Dict[str, Any]]


@dataclass
class ToolExecution:
    tool_name: str
    allowed: bool
    executed: bool
    reason: str
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ToolAdapterRegistry:
    def __init__(self) -> None:
        self._adapters: Dict[str, ToolAdapterFunc] = {}

    def register(self, tool_name: str, adapter: ToolAdapterFunc) -> None:
        self._adapters[tool_name] = adapter

    def has(self, tool_name: str) -> bool:
        return tool_name in self._adapters

    def list_tools(self) -> list[str]:
        return sorted(self._adapters.keys())

    def execute(self, request: ToolRequest, context: Dict[str, Any]) -> Dict[str, Any]:
        if request.tool_name not in self._adapters:
            raise KeyError(f"No adapter registered for tool '{request.tool_name}'")
        return self._adapters[request.tool_name](request, context)


def resolve_tool_output_path(*, output_dir: str, tenant_id: str, session_id: str, filename: str, create_parent: bool = True):
    """Return a proven-contained output path for any future side-effect adapter."""
    return resolve_contained_path(
        output_dir,
        str(tenant_id),
        str(session_id),
        str(filename),
        create_parent=create_parent,
    )


def extract_tool_output_text(output: Any, *, max_chars: int = 8000) -> str:
    """Best-effort textual projection of tool output for trust-boundary reingestion.

    The extraction is intentionally conservative: it only normalizes likely text
    carriers and avoids leaking oversized blobs into downstream checks.
    """

    def _clip(value: str) -> str:
        return str(value or "")[: max(256, int(max_chars))]

    if output is None:
        return ""
    if isinstance(output, str):
        return _clip(output)
    if isinstance(output, Mapping):
        # Prefer explicit textual fields first.
        for key in ("text", "summary", "content", "payload_preview", "message"):
            value = output.get(key)
            if isinstance(value, str) and value.strip():
                return _clip(value)
        # Then common nested structures.
        if isinstance(output.get("echo"), (dict, list)):
            return _clip(json.dumps(output.get("echo"), ensure_ascii=True, default=str))
        if isinstance(output.get("echo"), str):
            return _clip(str(output.get("echo")))
        return _clip(json.dumps(output, ensure_ascii=True, default=str))
    if isinstance(output, list):
        return _clip(json.dumps(output, ensure_ascii=True, default=str))
    return _clip(str(output))


def _summarize_adapter(request: ToolRequest, context: Dict[str, Any]) -> Dict[str, Any]:
    text = str(request.args.get("text") or request.args.get("raw_args") or "")
    if not text:
        allowed_items = context.get("allowed_items", [])
        text = "\n".join(getattr(item, "text", "") for item in allowed_items[:2])
    summary = text.strip().replace("\n", " ")[:280]
    return {"summary": summary, "chars": len(summary)}


def _retrieval_readonly_adapter(request: ToolRequest, context: Dict[str, Any]) -> Dict[str, Any]:
    allowed_items = context.get("allowed_items", [])
    return {
        "query": str(request.args.get("query") or request.args.get("raw_args") or ""),
        "doc_ids": [getattr(item, "doc_id", "") for item in allowed_items],
        "count": len(allowed_items),
    }


def _echo_adapter(request: ToolRequest, context: Dict[str, Any]) -> Dict[str, Any]:
    return {"echo": request.args}


def build_default_tool_registry() -> ToolAdapterRegistry:
    registry = ToolAdapterRegistry()
    registry.register("summarize", _summarize_adapter)
    registry.register("retrieval_readonly", _retrieval_readonly_adapter)
    registry.register("echo", _echo_adapter)
    return registry
