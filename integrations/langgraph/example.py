"""LangGraph integration minimal example.

This snippet demonstrates the canonical Omega guard wiring point.
"""

from __future__ import annotations

from omega.integrations import OmegaLangGraphGuard


def secure_graph(compiled_graph):
    guard = OmegaLangGraphGuard(profile="quickstart")
    return guard.wrap_graph(compiled_graph)


def main() -> None:
    print("See function definitions in this file and run the smoke command from README.")


if __name__ == "__main__":
    main()
