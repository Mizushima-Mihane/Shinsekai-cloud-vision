"""Abstract base for cloud vision API providers and a class-level registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Type


class BaseVisionProvider(ABC):
    """Send an image + text prompt to a cloud vision API and return a description."""

    def __init__(self, api_key: str = "", base_url: str = "", model: str = "", **kwargs: Any) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self._extra = kwargs

    @abstractmethod
    def describe_image(self, image_bytes: bytes, mime_type: str, prompt: str) -> str: ...

    @classmethod
    @abstractmethod
    def provider_id(cls) -> str:
        """Unique lower-case slug, e.g. ``"openai"``."""

    @classmethod
    @abstractmethod
    def display_name(cls) -> str:
        """Human-readable label shown in UI."""

    @classmethod
    def default_base_url(cls) -> str:
        """Default API endpoint when none is configured."""
        return ""

    @classmethod
    def default_model(cls) -> str:
        """Default model id for this provider."""
        return ""


# ── Class-level provider registry ──────────────────────────────────

class VisionProviderRegistry:
    """Registry of known :class:`BaseVisionProvider` subclasses.

    Populated at import time by each provider module — similar to
    ``LLMAdapterFactory._adapters``.
    """

    _providers: Dict[str, Type[BaseVisionProvider]] = {}

    @classmethod
    def register(cls, provider_cls: Type[BaseVisionProvider]) -> None:
        pid = provider_cls.provider_id()
        cls._providers[pid] = provider_cls

    @classmethod
    def get(cls, provider_id: str, **kwargs: Any) -> BaseVisionProvider:
        """Instantiate a provider by id; raises ``ValueError`` if unknown."""
        pid = (provider_id or "openai").strip().lower()
        cls_ = cls._providers.get(pid)
        if cls_ is None:
            known = ",".join(sorted(cls._providers))
            raise ValueError(
                f"Unknown vision provider '{pid}'. Known: {known}"
            )
        return cls_(**kwargs)

    @classmethod
    def list_providers(cls) -> Dict[str, str]:
        """Return ``{provider_id: display_name, ...}`` for all registered providers."""
        return {p.provider_id(): p.display_name() for p in cls._providers.values()}

    @classmethod
    def default_provider(cls) -> str:
        return next(iter(cls._providers), "openai")
