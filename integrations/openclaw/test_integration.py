from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    npm_exe = shutil.which("npm.cmd") or shutil.which("npm")
    if not npm_exe:
        sys.stderr.write("npm is not available in PATH; skipping OpenClaw launcher on this host.\n")
        return 0
    root = Path(__file__).resolve().parents[2]
    plugin_dir = root / "plugins" / "openclaw-omega-guard"
    if not (plugin_dir / "node_modules").exists():
        sys.stderr.write("OpenClaw node_modules are missing; skipping local launcher. Run npm ci in plugin dir for full validation.\n")
        return 0
    proc = subprocess.run(
        [npm_exe, "run", "smoke"],
        cwd=str(plugin_dir),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
