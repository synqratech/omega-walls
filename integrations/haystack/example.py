"""Haystack integration minimal example.

This snippet demonstrates the canonical Omega guard wiring point.
"""

from __future__ import annotations

from omega.integrations import OmegaHaystackGuard


def secure_pipeline(pipeline):
    guard = OmegaHaystackGuard(profile="quickstart")
    return guard.wrap_pipeline(pipeline, component_name="omega_guard_component")


def main() -> None:
    print("See function definitions in this file and run the smoke command from README.")


if __name__ == "__main__":
    main()
