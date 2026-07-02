"""智谱 GLM-4V vision provider — OpenAI-compatible."""

from __future__ import annotations

from plugins.cloud_vision.image_utils import data_uri, detect_mime_type
from plugins.cloud_vision.vision_provider import BaseVisionProvider, VisionProviderRegistry
from sdk.logging import get_logger

logger = get_logger(__name__, plugin_id="com.shinsekai.cloud_vision")


class ZhipuVisionProvider(BaseVisionProvider):
    """Describe images via 智谱 AI GLM-4V (OpenAI-compatible)."""

    @classmethod
    def provider_id(cls) -> str:
        return "zhipu"

    @classmethod
    def display_name(cls) -> str:
        return "智谱 GLM-4V"

    @classmethod
    def default_base_url(cls) -> str:
        return "https://open.bigmodel.cn/api/paas/v4"

    @classmethod
    def default_model(cls) -> str:
        return "glm-4v"

    def describe_image(self, image_bytes: bytes, mime_type: str, prompt: str) -> str:
        from openai import OpenAI

        api_key = self.api_key or _fallback_llm_key("智谱AI")
        base_url = self.base_url or self.default_base_url()
        model = self.model or self.default_model()

        if not api_key:
            return _NO_KEY.format(provider="GLM-4V")

        client = OpenAI(api_key=api_key, base_url=base_url)

        # 智谱 GLM-4V image format: {"type": "image_url", "image_url": {"url": "data:..."}}
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
            logger.debug("Zhipu vision API call failed: %s", exc)
            return _ERROR.format(provider="GLM-4V", error=exc)


_NO_KEY = "[Cloud Vision] {provider} API key 未配置，请在插件设置中填写。"
_ERROR = "[Cloud Vision] {provider} API 错误: {error}"


def _fallback_llm_key(provider: str) -> str:
    try:
        from config.config_manager import ConfigManager

        return (ConfigManager().config.api_config.llm_api_key or {}).get(provider, "")
    except Exception:
        return ""


VisionProviderRegistry.register(ZhipuVisionProvider)
