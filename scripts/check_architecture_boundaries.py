from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class BoundaryViolation:
    file: str
    line: int
    import_stmt: str
    reason: str


def _python_files(root: Path, rel_dir: str) -> Iterable[Path]:
    base = root / rel_dir
    if not base.exists():
        return []
    return [p for p in base.rglob("*.py") if p.is_file()]


def _import_name(node: ast.AST) -> List[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        if node.module:
            return [node.module]
    return []


def scan_boundaries(root: Path) -> List[BoundaryViolation]:
    violations: List[BoundaryViolation] = []
    for py_file in _python_files(root, "omega"):
        rel = py_file.relative_to(root).as_posix()
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=rel)
        except SyntaxError as exc:
            violations.append(
                BoundaryViolation(
                    file=rel,
                    line=int(exc.lineno or 0),
                    import_stmt="",
                    reason="syntax_error",
                )
            )
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for name in _import_name(node):
                top = str(name).split(".", 1)[0]
                if top == "enterprise":
                    violations.append(
                        BoundaryViolation(
                            file=rel,
                            line=int(getattr(node, "lineno", 0) or 0),
                            import_stmt=str(name),
                            reason="omega_must_not_import_enterprise",
                        )
                    )
                if top == "internal_data":
                    violations.append(
                        BoundaryViolation(
                            file=rel,
                            line=int(getattr(node, "lineno", 0) or 0),
                            import_stmt=str(name),
                            reason="omega_must_not_import_internal_data",
                        )
                    )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description="Check architecture boundaries between core and enterprise layers.")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    violations = scan_boundaries(root)
    payload = {
        "event": "architecture_boundary_check_v1",
        "root": str(root.as_posix()),
        "violations_total": int(len(violations)),
        "violations": [
            {
                "file": v.file,
                "line": int(v.line),
                "import": v.import_stmt,
                "reason": v.reason,
            }
            for v in violations
        ],
        "status": "ok" if not violations else "violation",
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    if args.strict and violations:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
