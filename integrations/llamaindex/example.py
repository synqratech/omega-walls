"""LlamaIndex integration minimal example.

This snippet demonstrates the canonical Omega guard wiring point.
"""

from __future__ import annotations

from omega.integrations import OmegaLlamaIndexGuard


def secure_query_engine(index):
    guard = OmegaLlamaIndexGuard(profile="quickstart")
    return guard.wrap_query_engine(index.as_query_engine())


def main() -> None:
    print("See function definitions in this file and run the smoke command from README.")


if __name__ == "__main__":
    main()
