"""Tool adapter registry and default adapters for local execution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from omega.interfaces.contracts_v1 import ToolRequest


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

    def execute(self, request: ToolRequest, context: Dict[str, Any]) -> Dict[str, Any]:
        if request.tool_name not in self._adapters:
            raise KeyError(f"No adapter registered for tool '{request.tool_name}'")
        return self._adapters[request.tool_name](request, context)


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


def _write_file_adapter(request: ToolRequest, context: Dict[str, Any]) -> Dict[str, Any]:
    root = Path(context.get("tool_output_dir", "artifacts/tools"))
    session_id = str(context.get("session_id", request.session_id))
    step = int(context.get("step", request.step))
    session_dir = root / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    requested_name = str(request.args.get("filename") or f"{request.tool_name}_{step}.txt")
    safe_name = Path(requested_name).name
    path = session_dir / safe_name

    content = request.args.get("content")
    if content is None:
        content = request.args.get("raw_args", "")
    content_str = str(content)
    path.write_text(content_str, encoding="utf-8")

    return {"path": str(path.resolve()), "bytes": len(content_str)}


def _network_post_adapter(request: ToolRequest, context: Dict[str, Any]) -> Dict[str, Any]:
    # Local smoke: no real egress, but full adapter path is exercised.
    url = str(request.args.get("url") or "https://example.invalid")
    payload = request.args.get("payload")
    if payload is None:
        payload = request.args.get("raw_args", "")
    payload_preview = str(payload)[:200]
    return {
        "status": "dry_run",
        "url": url,
        "payload_preview": payload_preview,
    }


def _echo_adapter(request: ToolRequest, context: Dict[str, Any]) -> Dict[str, Any]:
    return {"echo": request.args}


def build_default_tool_registry() -> ToolAdapterRegistry:
    registry = ToolAdapterRegistry()
    registry.register("summarize", _summarize_adapter)
    registry.register("retrieval_readonly", _retrieval_readonly_adapter)
    registry.register("write_file", _write_file_adapter)
    registry.register("network_post", _network_post_adapter)
    registry.register("echo", _echo_adapter)
    return registry
