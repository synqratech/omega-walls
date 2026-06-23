"""Provider client adapters for APIPerceptionProjector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, TYPE_CHECKING

from omega.interfaces.contracts_v1 import WALLS_V1
from omega.projector.api_hybrid.semantic_contracts import (
    ProviderCapabilities,
    ProviderSemanticResponse,
    SemanticInput,
    SemanticResult,
    SemanticTextPart,
)

if TYPE_CHECKING:
    from omega.projector.api_hybrid_projector import APIPerceptionProjector


_IMAGE_MIMES = ("image/gif", "image/jpeg", "image/png", "image/webp")


def capabilities_for_provider(
    provider: str,
    configured: Mapping[str, Any] | None = None,
) -> ProviderCapabilities:
    name = str(provider or "").strip().lower()
    raw = dict(configured or {})
    built_in_image = name in {"openai", "anthropic", "local_vision"}
    text = raw.get("text", True)
    image = raw.get("image", built_in_image)
    if type(text) is not bool or type(image) is not bool:
        raise ValueError("provider capabilities text/image must be boolean")
    if image and name not in {"openai", "anthropic", "openai_compat", "local_vision"}:
        raise ValueError(f"provider {name} does not implement image semantic support")
    supported = raw.get("supported_image_mime_types", _IMAGE_MIMES if image else ())
    if not isinstance(supported, (list, tuple)):
        raise ValueError("supported_image_mime_types must be a list")
    max_image_bytes = int(raw.get("max_image_bytes", 20 * 1024 * 1024))
    max_images = int(raw.get("max_images", 8 if image else 1))
    return ProviderCapabilities(
        text=text,
        image=image,
        supported_image_mime_types=tuple(str(x) for x in supported),
        max_image_bytes=max_image_bytes,
        max_images=max_images,
    )


def _unsupported_result(
    *, provider: str, capabilities: ProviderCapabilities
) -> ProviderSemanticResponse:
    return ProviderSemanticResponse(
        result=SemanticResult(
            pressure_signed={str(w): 0.0 for w in WALLS_V1},
            directive_intent={str(w): False for w in WALLS_V1},
            defensive_context=False,
            confidence=0.0,
            semantic_status="vision_unsupported",
            provider_meta={
                "provider": str(provider),
                "capabilities": capabilities.to_dict(),
            },
            vision_meta={"attempted": True, "provider_supported": False},
        ),
        response_id="",
        retries_used=0,
    )


class ProviderClient:
    provider: str
    capabilities: ProviderCapabilities

    def score_semantic(
        self,
        *,
        semantic_input: SemanticInput,
        system_prompt: str,
        user_prompt: str,
        model: str,
        timeout_sec: float,
        retries: int,
        metadata: Mapping[str, Any],
    ) -> ProviderSemanticResponse:
        raise NotImplementedError

    def score_text(
        self,
        *,
        text: str,
        system_prompt: str,
        user_prompt: str,
        model: str,
        timeout_sec: float,
        retries: int,
        metadata: Mapping[str, Any],
    ) -> ProviderSemanticResponse:
        return self.score_semantic(
            semantic_input=SemanticInput(
                text_parts=(SemanticTextPart(text=str(text or "")),)
            ),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            timeout_sec=timeout_sec,
            retries=retries,
            metadata=metadata,
        )

    def _wrap_raw(
        self,
        *,
        raw: tuple[Dict[str, Any], str, int],
        semantic_input: SemanticInput,
        provider_id: str | None = None,
    ) -> ProviderSemanticResponse:
        payload, response_id, retries_used = raw
        vision = bool(semantic_input.image_parts)
        result = SemanticResult.from_payload(
            payload=payload,
            semantic_status=("vision_semantic_active" if vision else "semantic_active"),
            provider_meta={
                "provider": self.provider,
                "provider_id": str(provider_id or self.provider),
                "capabilities": self.capabilities.to_dict(),
            },
            vision_meta={
                "attempted": vision,
                "provider_supported": bool(self.capabilities.image),
            },
        )
        return ProviderSemanticResponse(
            result=result, response_id=response_id, retries_used=retries_used
        )


@dataclass
class OpenAIProviderClient(ProviderClient):
    projector: "APIPerceptionProjector"
    capabilities: ProviderCapabilities
    provider: str = "openai"

    def score_semantic(
        self,
        *,
        semantic_input: SemanticInput,
        system_prompt: str,
        user_prompt: str,
        model: str,
        timeout_sec: float,
        retries: int,
        metadata: Mapping[str, Any],
    ) -> ProviderSemanticResponse:
        _ = (model, timeout_sec, retries)
        if not self.capabilities.supports_input(semantic_input):
            return _unsupported_result(
                provider=self.provider, capabilities=self.capabilities
            )
        raw = self.projector._call_openai_provider_scores(
            semantic_input=semantic_input,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            use_responses=True,
            metadata=metadata,
        )
        return self._wrap_raw(raw=raw, semantic_input=semantic_input)


@dataclass
class OpenAICompatProviderClient(ProviderClient):
    projector: "APIPerceptionProjector"
    capabilities: ProviderCapabilities
    provider: str = "openai_compat"

    def score_semantic(
        self,
        *,
        semantic_input: SemanticInput,
        system_prompt: str,
        user_prompt: str,
        model: str,
        timeout_sec: float,
        retries: int,
        metadata: Mapping[str, Any],
    ) -> ProviderSemanticResponse:
        _ = (model, timeout_sec, retries)
        if not self.capabilities.supports_input(semantic_input):
            return _unsupported_result(
                provider=self.provider, capabilities=self.capabilities
            )
        raw = self.projector._call_openai_provider_scores(
            semantic_input=semantic_input,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            use_responses=False,
            metadata=metadata,
        )
        return self._wrap_raw(raw=raw, semantic_input=semantic_input)


@dataclass
class AnthropicProviderClient(ProviderClient):
    projector: "APIPerceptionProjector"
    capabilities: ProviderCapabilities
    provider: str = "anthropic"

    def score_semantic(
        self,
        *,
        semantic_input: SemanticInput,
        system_prompt: str,
        user_prompt: str,
        model: str,
        timeout_sec: float,
        retries: int,
        metadata: Mapping[str, Any],
    ) -> ProviderSemanticResponse:
        _ = (model, timeout_sec, retries)
        if not self.capabilities.supports_input(semantic_input):
            return _unsupported_result(
                provider=self.provider, capabilities=self.capabilities
            )
        raw = self.projector._call_anthropic_provider_scores(
            semantic_input=semantic_input,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            metadata=metadata,
        )
        return self._wrap_raw(raw=raw, semantic_input=semantic_input)


@dataclass
class LocalVisionProviderClient(ProviderClient):
    projector: "APIPerceptionProjector"
    capabilities: ProviderCapabilities
    provider: str = "local_vision"

    def score_semantic(
        self,
        *,
        semantic_input: SemanticInput,
        system_prompt: str,
        user_prompt: str,
        model: str,
        timeout_sec: float,
        retries: int,
        metadata: Mapping[str, Any],
    ) -> ProviderSemanticResponse:
        _ = (system_prompt, user_prompt, model, timeout_sec, retries, metadata)
        if not self.capabilities.supports_input(semantic_input):
            return _unsupported_result(
                provider=self.provider, capabilities=self.capabilities
            )
        raw = self.projector._call_local_vision_scores(semantic_input=semantic_input)
        return self._wrap_raw(raw=raw, semantic_input=semantic_input)


def build_provider_client(
    *,
    projector: "APIPerceptionProjector",
    provider: str,
    capabilities: Mapping[str, Any] | ProviderCapabilities | None = None,
) -> ProviderClient:
    p = str(provider or "").strip().lower()
    resolved = (
        capabilities
        if isinstance(capabilities, ProviderCapabilities)
        else capabilities_for_provider(p, capabilities)
    )
    if p == "anthropic":
        return AnthropicProviderClient(projector=projector, capabilities=resolved)
    if p == "openai_compat":
        return OpenAICompatProviderClient(projector=projector, capabilities=resolved)
    if p == "local_vision":
        return LocalVisionProviderClient(projector=projector, capabilities=resolved)
    return OpenAIProviderClient(projector=projector, capabilities=resolved)
