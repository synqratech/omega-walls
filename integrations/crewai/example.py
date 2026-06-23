"""CrewAI integration minimal example.

This snippet demonstrates the canonical Omega guard wiring point.
"""

from __future__ import annotations

from omega.integrations import OmegaCrewAIGuard


def run_secure_crew(crew, inputs):
    guard = OmegaCrewAIGuard(profile="quickstart")
    with guard.install_global_hooks():
        return crew.kickoff(inputs=inputs)


def main() -> None:
    print("See function definitions in this file and run the smoke command from README.")


if __name__ == "__main__":
    main()
