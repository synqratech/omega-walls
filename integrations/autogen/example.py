"""AutoGen integration minimal example.

This snippet demonstrates the canonical Omega guard wiring point.
"""

from __future__ import annotations

from omega.integrations import OmegaAutoGenGuard


def secure_agent(agent):
    guard = OmegaAutoGenGuard(profile="quickstart")
    return guard.wrap_agent(agent)


def main() -> None:
    print("See function definitions in this file and run the smoke command from README.")


if __name__ == "__main__":
    main()
