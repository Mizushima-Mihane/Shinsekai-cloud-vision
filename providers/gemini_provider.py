"""Google Gemini Vision provider."""

from __future__ import annotations

from plugins.cloud_vision.vision_provider import BaseVisionProvider, VisionProviderRegistry
from plugins.cloud_vision.image_utils import detect_mime_type
from sdk.logging import get_logger

logger = get_logger(__name__, plugin_id="com.shinsekai.cloud_vision")


class GeminiVisionProvider(BaseVisionProvider):
    """Describe images via Google Gemini generate-content with inline image parts."""

    @classmethod
    def provider_id(cls) -> str:
        return "gemini"

    @classmethod
    def display_name(cls) -> str:
        return "Gemini Vision (Google)"

    @classmethod
    def default_model(cls) -> str:
        return "gemini-2.0-flash"

    def describe_image(self, image_bytes: bytes, mime_type: str, prompt: str) -> str:
        api_key = self.api_key or _fallback_api_key("Gemini")
        model = self.model or self.default_model()

        if not api_key:
            return _NO_KEY_MESSAGE.format(provider="Gemini")

        import google.genai as genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        mime = mime_type or detect_mime_type(image_bytes)

        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_bytes(data=image_bytes, mime_type=mime),
                            types.Part.from_text(text=prompt),
                        ],
                    )
                ],
            )
            return response.text or ""
        except Exception as exc:
            logger.debug("Gemini vision API call failed: %s", exc)
            return _ERROR_MESSAGE.format(provider="Gemini", error=exc)


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


VisionProviderRegistry.register(GeminiVisionProvider)
