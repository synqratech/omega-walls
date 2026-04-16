"""Public exception model for Omega SDK integrations."""

from __future__ import annotations

from typing import Any, Optional


class OmegaSDKError(RuntimeError):
    """Base class for user-facing SDK errors."""


class OmegaMissingDependencyError(OmegaSDKError):
    """Raised when a required dependency is not installed."""

    def __init__(self, dependency: str, *, hint: Optional[str] = None):
        self.dependency = str(dependency or "unknown")
        text = f"Missing dependency: {self.dependency}"
        if hint:
            text = f"{text}. {hint}"
        super().__init__(text)


class OmegaConfigError(OmegaSDKError):
    """Raised when configuration cannot be loaded or validated."""


class OmegaAPIError(OmegaSDKError):
    """Raised when API-backed perception fails or is misconfigured."""

    def __init__(self, message: str, *, code: Optional[int] = None, details: Optional[Any] = None):
        self.code = int(code) if code is not None else None
        self.details = details
        super().__init__(str(message))


class OmegaInitializationError(OmegaSDKError):
    """Raised when SDK runtime components fail to initialize."""


class OmegaRuntimeError(OmegaSDKError):
    """Raised when detection step execution fails at runtime."""

