"""OpenAI Vision provider (GPT-4o / GPT-4V)."""

from __future__ import annotations

from plugins.cloud_vision.vision_provider import BaseVisionProvider, VisionProviderRegistry
from plugins.cloud_vision.image_utils import data_uri, detect_mime_type
from sdk.logging import get_logger

logger = get_logger(__name__, plugin_id="com.shinsekai.cloud_vision")


class OpenAIVisionProvider(BaseVisionProvider):
    """Describe images via OpenAI chat-completion with image_url content blocks."""

    @classmethod
    def provider_id(cls) -> str:
        return "openai"

    @classmethod
    def display_name(cls) -> str:
        return "OpenAI Vision (GPT-4o)"

    @classmethod
    def default_base_url(cls) -> str:
        return "https://api.openai.com/v1"

    @classmethod
    def default_model(cls) -> str:
        return "gpt-4o"

    def describe_image(self, image_bytes: bytes, mime_type: str, prompt: str) -> str:
        from openai import OpenAI

        api_key = self.api_key or _fallback_api_key("ChatGPT")
        base_url = self.base_url or self.default_base_url()
        model = self.model or self.default_model()

        if not api_key:
            return _NO_KEY_MESSAGE.format(provider="OpenAI")

        client = OpenAI(api_key=api_key, base_url=base_url)
        data_url = data_uri(image_bytes, mime_type or detect_mime_type(image_bytes))

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url, "detail": "auto"}},
                        ],
                    }
                ],
                max_tokens=1024,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.debug("OpenAI vision API call failed: %s", exc)
            return _ERROR_MESSAGE.format(provider="OpenAI", error=exc)


# ── helpers ────────────────────────────────────────────────────────

_NO_KEY_MESSAGE = (
    "[Cloud Vision] {provider} API key is not configured. "
    "Please set the key in the plugin settings page."
)

_ERROR_MESSAGE = (
    "[Cloud Vision] {provider} API error: {error}"
)


def _fallback_api_key(provider_name: str) -> str:
    """Read the global LLM API key for *provider_name* from ``api.yaml``."""
    try:
        from config.config_manager import ConfigManager

        cm = ConfigManager()
        return (cm.config.api_config.llm_api_key or {}).get(provider_name, "")
    except Exception:
        return ""


VisionProviderRegistry.register(OpenAIVisionProvider)
