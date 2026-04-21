from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace
from typing import Any, Callable, Dict, List

import pytest

from omega.adapters import OmegaToolBlockedError
from omega.integrations.autogen_guard import OmegaAutoGenGuard
from omega.integrations.crewai_guard import OmegaCrewAIGuard
from omega.integrations.haystack_guard import OmegaHaystackGuard
from omega.integrations.langchain_guard import OmegaLangChainGuard
from omega.integrations.langgraph_guard import OmegaLangGraphGuard
from omega.integrations.llamaindex_guard import OmegaLlamaIndexGuard


def _langchain_call(guard: OmegaLangChainGuard, blocked: bool, idx: int, counter: Dict[str, int]) -> str:
    tool_name = "unknown_tool_smoke" if blocked else "echo"
    tool_args = {"payload": f"p-{idx}"}
    request = SimpleNamespace(
        tool_call={"name": tool_name, "args": tool_args},
        state={"thread_id": f"lc-race-{idx}"},
        runtime=None,
    )

    def _handler(_req: Any) -> Dict[str, Any]:
        counter["calls"] += 1
        return {"ok": True}

    try:
        guard._wrap_tool_call_impl(request=request, handler=_handler)  # noqa: SLF001
        return "allow"
    except OmegaToolBlockedError:
        return "blocked"


def _wrapped_tool_runner(
    wrap_tool: Callable[[str, Callable[..., Any]], Callable[..., Any]],
    *,
    blocked: bool,
    idx: int,
    counter: Dict[str, int],
) -> str:
    tool_name = "unknown_tool_smoke" if blocked else "echo"

    def _echo(payload: str, **kwargs: Any) -> str:
        del kwargs
        counter["calls"] += 1
        return f"echo:{payload}"

    wrapped = wrap_tool(tool_name, _echo)
    try:
        _ = wrapped(f"p-{idx}", thread_id=f"race-{idx}", actor_id=f"user-{idx}")
        return "allow"
    except OmegaToolBlockedError:
        return "blocked"


@pytest.mark.parametrize(
    ("name", "factory", "runner"),
    [
        ("langchain_guard", lambda: OmegaLangChainGuard(profile="dev", projector_mode="pi0"), _langchain_call),
        (
            "langgraph_guard",
            lambda: OmegaLangGraphGuard(profile="dev", projector_mode="pi0"),
            lambda g, blocked, idx, counter: _wrapped_tool_runner(g.wrap_tool, blocked=blocked, idx=idx, counter=counter),
        ),
        (
            "llamaindex_guard",
            lambda: OmegaLlamaIndexGuard(profile="dev", projector_mode="pi0"),
            lambda g, blocked, idx, counter: _wrapped_tool_runner(g.wrap_tool, blocked=blocked, idx=idx, counter=counter),
        ),
        (
            "haystack_guard",
            lambda: OmegaHaystackGuard(profile="dev", projector_mode="pi0"),
            lambda g, blocked, idx, counter: _wrapped_tool_runner(g.wrap_tool, blocked=blocked, idx=idx, counter=counter),
        ),
        (
            "autogen_guard",
            lambda: OmegaAutoGenGuard(profile="dev", projector_mode="pi0"),
            lambda g, blocked, idx, counter: _wrapped_tool_runner(g.wrap_tool, blocked=blocked, idx=idx, counter=counter),
        ),
        (
            "crewai_guard",
            lambda: OmegaCrewAIGuard(profile="dev", projector_mode="pi0"),
            lambda g, blocked, idx, counter: _wrapped_tool_runner(g.wrap_tool, blocked=blocked, idx=idx, counter=counter),
        ),
    ],
)
def test_framework_tool_race_fail_closed(name: str, factory: Callable[[], Any], runner: Callable[..., str]) -> None:
    guard = factory()
    counter = {"calls": 0}
    tasks: List[tuple[bool, int]] = []
    # Mixed load: allow and blocked paths in parallel.
    for idx in range(20):
        tasks.append((False, idx))
    for idx in range(20, 40):
        tasks.append((True, idx))

    outcomes: List[str] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(runner, guard, blocked, idx, counter) for blocked, idx in tasks]
        for fut in as_completed(futures):
            outcomes.append(str(fut.result()))

    assert outcomes.count("allow") == 20, f"{name}: expected 20 allow outcomes"
    assert outcomes.count("blocked") == 20, f"{name}: expected 20 blocked outcomes"
    # Downstream callable must run only for allowed path => no orphan executions.
    assert counter["calls"] == 20, f"{name}: downstream call mismatch under race"
