"""LangChain integration minimal example.

This snippet demonstrates the canonical Omega guard wiring point.
"""

from __future__ import annotations

from omega.integrations import OmegaLangChainGuard


def build_agent(create_agent, tools):
    guard = OmegaLangChainGuard(profile="quickstart")
    return create_agent(model="openai:gpt-5.4-mini", tools=tools, middleware=guard.middleware())


def main() -> None:
    print("See function definitions in this file and run the smoke command from README.")


if __name__ == "__main__":
    main()
