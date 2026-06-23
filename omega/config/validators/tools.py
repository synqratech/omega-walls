from __future__ import annotations

from typing import Any, Dict


_BUILTIN_DISABLED_TOOLS = frozenset({"write_file", "network_post"})


def validate_tools_config(config: Dict[str, Any]) -> None:
    tools_cfg = config.get("tools", {})
    execution_mode = str(tools_cfg.get("execution_mode", "ENFORCE")).upper()
    if execution_mode not in {"ENFORCE", "DRY_RUN"}:
        raise ValueError("tools.execution_mode must be ENFORCE or DRY_RUN")
    freeze_cfg = tools_cfg.get("freeze", {}) or {}
    if freeze_cfg and not isinstance(freeze_cfg, dict):
        raise ValueError("tools.freeze must be a mapping")
    if bool(freeze_cfg.get("read_only_exception", False)):
        raise ValueError("tools.freeze.read_only_exception must be false; use TOOLS_ALLOWLIST")
    capabilities = tools_cfg.get("capabilities", {}) or {}
    if capabilities and not isinstance(capabilities, dict):
        raise ValueError("tools.capabilities must be a mapping")
    for tool_name, capability in capabilities.items():
        if not isinstance(capability, dict):
            raise ValueError(f"tools.capabilities.{tool_name} must be a mapping")
        enabled = bool(capability.get("enabled", True))
        if str(tool_name).strip().lower() in _BUILTIN_DISABLED_TOOLS and enabled:
            raise ValueError(
                f"tools.capabilities.{tool_name}.enabled must be false; "
                "legacy side-effecting built-ins are not available"
            )
    arg_validation_cfg = tools_cfg.get("arg_validation", {}) or {}
    if arg_validation_cfg and not isinstance(arg_validation_cfg, dict):
        raise ValueError("tools.arg_validation must be a mapping")
    if isinstance(arg_validation_cfg, dict) and arg_validation_cfg:
        _ = bool(arg_validation_cfg.get("enabled", True))
        fail_mode = str(arg_validation_cfg.get("fail_mode", "deny")).strip().lower()
        if fail_mode != "deny":
            raise ValueError("tools.arg_validation.fail_mode must be deny")
        shell_patterns = arg_validation_cfg.get("shell_like_name_patterns", [])
        if shell_patterns is not None and not isinstance(shell_patterns, list):
            raise ValueError("tools.arg_validation.shell_like_name_patterns must be a list")
        net_cfg = arg_validation_cfg.get("network_post", {}) or {}
        wr_cfg = arg_validation_cfg.get("write_file", {}) or {}
        sh_cfg = arg_validation_cfg.get("shell_like", {}) or {}
        for key in ("max_payload_bytes", "max_headers", "max_header_key_chars", "max_header_value_chars"):
            if int(net_cfg.get(key, 1)) <= 0:
                raise ValueError(f"tools.arg_validation.network_post.{key} must be > 0")
        for key in ("max_filename_chars", "max_content_bytes"):
            if int(wr_cfg.get(key, 1)) <= 0:
                raise ValueError(f"tools.arg_validation.write_file.{key} must be > 0")
        if int(sh_cfg.get("max_command_chars", 1)) <= 0:
            raise ValueError("tools.arg_validation.shell_like.max_command_chars must be > 0")
        destructive_patterns = sh_cfg.get("destructive_patterns", [])
        if destructive_patterns is not None and not isinstance(destructive_patterns, list):
            raise ValueError("tools.arg_validation.shell_like.destructive_patterns must be a list")

