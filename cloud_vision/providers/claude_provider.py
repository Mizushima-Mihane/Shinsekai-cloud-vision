"""Anthropic Claude Vision provider."""

from __future__ import annotations

from plugins.cloud_vision.vision_provider import BaseVisionProvider, VisionProviderRegistry
from plugins.cloud_vision.image_utils import encode_base64, detect_mime_type
from sdk.logging import get_logger

logger = get_logger(__name__, plugin_id="com.shinsekai.cloud_vision")


class ClaudeVisionProvider(BaseVisionProvider):
    """Describe images via Anthropic Claude Messages API with image source blocks."""

    @classmethod
    def provider_id(cls) -> str:
        return "claude"

    @classmethod
    def display_name(cls) -> str:
        return "Claude Vision (Anthropic)"

    @classmethod
    def default_model(cls) -> str:
        return "claude-3-5-sonnet-20240620"

    def describe_image(self, image_bytes: bytes, mime_type: str, prompt: str) -> str:
        api_key = self.api_key or _fallback_api_key("Claude")
        model = self.model or self.default_model()

        if not api_key:
            return _NO_KEY_MESSAGE.format(provider="Claude")

        import anthropic

        client_kwargs: dict = {"api_key": api_key}
        if self.base_url:
            # Normalize base URL for the Anthropic SDK
            from llm.llm_adapter import normalize_claude_base_url_for_sdk

            client_kwargs["base_url"] = normalize_claude_base_url_for_sdk(self.base_url)

        client = anthropic.Anthropic(**client_kwargs)
        mime = mime_type or detect_mime_type(image_bytes)
        b64 = encode_base64(image_bytes)

        try:
            message = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime,
                                    "data": b64,
                                },
                            },
                        ],
                    }
                ],
            )
            # Extract text from the first text block
            for block in message.content:
                if getattr(block, "type", None) == "text":
                    return block.text
            return ""
        except Exception as exc:
            logger.debug("Claude vision API call failed: %s", exc)
            return _ERROR_MESSAGE.format(provider="Claude", error=exc)


_NO_KEY_MESSAGE = (
    "[Cloud Vision] {provider} API key is not configured. "
    "Please set the key in the plugin settings page."
)

_ERROR_MESSAGE = (
    "[Cloud Vision] {provider} API error: {error}"
)


def _fallback_api_key(provider_name: str) -> str:
    try:
        from config.config_manager import ConfigManager

        cm = ConfigManager()
        return (cm.config.api_config.llm_api_key or {}).get(provider_name, "")
    except Exception:
        return ""


VisionProviderRegistry.register(ClaudeVisionProvider)
