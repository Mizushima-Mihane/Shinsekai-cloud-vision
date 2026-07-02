"""Generic OpenAI-compatible vision provider — works with any OpenAI-format API."""

from __future__ import annotations

from plugins.cloud_vision.image_utils import data_uri, detect_mime_type
from plugins.cloud_vision.vision_provider import BaseVisionProvider, VisionProviderRegistry
from sdk.logging import get_logger

logger = get_logger(__name__, plugin_id="com.shinsekai.cloud_vision")


class OpenAICompatibleProvider(BaseVisionProvider):
    """Generic provider for any OpenAI-compatible vision API.

    Set the base URL and model ID in the plugin settings.
    Works with: 通义千问, 智谱, 豆包, DeepSeek-Vision, vLLM, Ollama, etc.
    """

    @classmethod
    def provider_id(cls) -> str:
        return "openai_compatible"

    @classmethod
    def display_name(cls) -> str:
        return "OpenAI 兼容（通用）"

    @classmethod
    def default_base_url(cls) -> str:
        return "https://api.openai.com/v1"

    @classmethod
    def default_model(cls) -> str:
        return "gpt-4o"

    def describe_image(self, image_bytes: bytes, mime_type: str, prompt: str) -> str:
        from openai import OpenAI

        api_key = self.api_key
        base_url = self.base_url or self.default_base_url()
        model = self.model or self.default_model()

        if not api_key:
            return "[Cloud Vision] API key 未配置，请在插件设置中填写。"

        client = OpenAI(api_key=api_key, base_url=base_url)
        data_url = data_uri(image_bytes, mime_type or detect_mime_type(image_bytes))

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }],
                max_tokens=1024,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.debug("Vision API call failed: %s", exc)
            return f"[Cloud Vision] API 错误: {exc}"


VisionProviderRegistry.register(OpenAICompatibleProvider)
