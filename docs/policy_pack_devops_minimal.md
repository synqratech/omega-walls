# DevOps Minimal Policy Pack (`devops_minimal`)

This profile is a thin policy/config override for DevOps-style agent workflows.

## What it does

- Hard deny:
  - destructive shell commands (for example `rm -rf`, `del /f /q`, `format`, `mkfs`, `dd if=`)
  - explicit shell exfil commands (`curl|wget` + secret/token credential markers + external URL)
- Require approval:
  - `git push --force`
  - production deploy commands (for example `kubectl apply -f prod.yaml`)

Implementation path is unchanged:
- `OffPolicy + ToolGateway + tools.arg_validation`

## Run with profile

Use `devops_minimal` as your runtime profile:

```bash
omega-walls --profile devops_minimal --text "run ops command" --query "ops"
```

Or in code:

```python
from omega.config.loader import load_resolved_config
cfg = load_resolved_config(profile="devops_minimal").resolved
```

## Action mapping

- Hard deny from arg validation:
  - `INVALID_TOOL_ARGS_SHELLLIKE`
- Approval-gated tool paths:
  - `HUMAN_APPROVAL_REQUIRED`
  - `REQUIRE_APPROVAL_PENDING`

If a command is allowed after explicit approval (`human_approved=true`) and no deny pattern matches, tool execution can proceed.

## Notes

- This pack adds shell-like capabilities in profile only (`shell_exec`, `git_exec`, `deploy_exec`, etc.) without core refactors.
- If your tool names differ, extend:
  - `tools.capabilities.*`
  - `tools.arg_validation.shell_like_name_patterns`
