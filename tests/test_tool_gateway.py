from __future__ import annotations

from omega.interfaces.contracts_v1 import OffAction, ToolRequest


def test_gateway_freeze_disabled_mode(gateway):
    req = ToolRequest(tool_name="network_post", args={}, session_id="s", step=1)
    action = OffAction(type="TOOL_FREEZE", target="TOOLS", tool_mode="TOOLS_DISABLED", horizon_steps=20)
    dec = gateway.enforce(req, [action])
    assert dec.allowed is False
    assert dec.reason == "TOOL_FREEZE_ACTIVE"


def test_gateway_allowlist(gateway):
    req_allowed = ToolRequest(tool_name="summarize", args={}, session_id="s", step=1)
    req_blocked = ToolRequest(tool_name="network_post", args={}, session_id="s", step=1)
    action = OffAction(type="TOOL_FREEZE", target="TOOLS", tool_mode="TOOLS_ALLOWLIST", allowlist=["summarize"])

    dec_allowed = gateway.enforce(req_allowed, [action])
    dec_blocked = gateway.enforce(req_blocked, [action])

    assert dec_allowed.allowed is True
    assert dec_blocked.allowed is False
    assert dec_blocked.reason == "NOT_IN_ALLOWLIST"


def test_gateway_requires_human_approval_for_dangerous_tools(gateway):
    req_no_approval = ToolRequest(tool_name="network_post", args={}, session_id="s", step=1)
    req_with_approval = ToolRequest(
        tool_name="network_post",
        args={"human_approved": True},
        session_id="s",
        step=1,
    )

    dec_no = gateway.enforce(req_no_approval, [])
    dec_yes = gateway.enforce(req_with_approval, [])

    assert dec_no.allowed is False
    assert dec_no.reason == "HUMAN_APPROVAL_REQUIRED"
    assert dec_yes.allowed is True


def test_gateway_freeze_allows_read_only_exception(gateway):
    action = OffAction(type="TOOL_FREEZE", target="TOOLS", tool_mode="TOOLS_DISABLED", horizon_steps=5)
    req = ToolRequest(tool_name="summarize", args={}, session_id="s", step=1)
    dec = gateway.enforce(req, [action])
    assert dec.allowed is True
