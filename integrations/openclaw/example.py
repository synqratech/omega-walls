"""OpenClaw integration minimal smoke bridge example.

This script only demonstrates environment contract values used by the plugin smoke.
"""

from __future__ import annotations

import os


def main() -> None:
    required = [
        "OMEGA_OPENCLAW_API_BASE_URL",
        "OMEGA_OPENCLAW_API_KEY",
        "OMEGA_OPENCLAW_HMAC_SECRET",
    ]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        print({"status": "warn", "missing_env": missing})
    else:
        print({"status": "ok", "message": "OpenClaw local API smoke env is configured."})


if __name__ == "__main__":
    main()
