"""before_chat_hook: handles stashed screenshots (tool Mode 2) and stashed descriptions (auto/proactive monitors)."""

from __future__ import annotations

import time
from typing import Callable, List, Optional

from plugins.cloud_vision.config_model import CloudVisionConfig
from plugins.cloud_vision.image_utils import compress_if_needed, data_uri, detect_mime_type, encode_base64
from plugins.cloud_vision.screen_tool import pop_stashed_description, pop_stashed_screenshot
from sdk.hooks import BeforeChatContext
from sdk.logging import get_logger

logger = get_logger(__name__, plugin_id="com.shinsekai.cloud_vision")

_TRIGGER_TEXT = "[Screen]"


def _get_llm_provider() -> str:
    try:
        from config.config_manager import ConfigManager
        cm = ConfigManager()
        return (cm.config.api_config.llm_provider or "ChatGPT").strip()
    except Exception:
        return "ChatGPT"


def create_screen_hook(cfg: CloudVisionConfig) -> Callable[[BeforeChatContext], None]:
    """Build a ``before_chat_hook`` that:
    1. Injects stashed screenshots from the tool (Mode 2).
    2. Injects stashed descriptions from auto/proactive monitors into system prompt,
       and removes the ``[Screen]`` trigger from history.
    """

    def hook(context: BeforeChatContext) -> None:
        # ── Handle stashed description (auto/proactive monitors) ───
        desc = pop_stashed_description()
        if desc:
            _inject_system_prefix(
                context,
                f"[Screen]\n{desc}\n[/Screen]\n\n",
            )
            _remove_trigger(context)

        # ── Handle stashed screenshot (tool Mode 2) ────────────────
        stashed, stashed_query = pop_stashed_screenshot()
        if stashed is None:
            return

        image_bytes = compress_if_needed(stashed, cfg.max_image_size_mb)
        logger.info("Injecting stashed screenshot from tool: size=%d multimodal=%s",
                    len(image_bytes), cfg.multimodal_detected)

        if cfg.multimodal_detected:
            _inject_multimodal(context, image_bytes, stashed_query)
        else:
            _inject_description(context, image_bytes, stashed_query, cfg)

    return hook


def _remove_trigger(context: BeforeChatContext) -> None:
    """Remove ``[Screen]`` trigger messages from the message list so they
    don't occupy history. Also removes the last user message if it's just
    the trigger text."""
    messages: List[dict] = context.messages
    # Remove trigger-only user messages from the end
    kept: List[dict] = []
    for m in messages:
        if m.get("role") == "user" and m.get("content", "").strip() == _TRIGGER_TEXT:
            continue
        kept.append(m)
    if len(kept) < len(messages):
        context.messages = kept


def _inject_system_prefix(context: BeforeChatContext, prefix: str) -> None:
    """Replace any existing ``[Screen]…[/Screen]`` block in the system prompt
    with the new *prefix*, avoiding context bloat from accumulated descriptions."""
    messages: List[dict] = context.messages
    for m in messages:
        if m.get("role") == "system":
            content = m.get("content") or ""
            # Replace existing [Screen] block
            if "[Screen]" in content and "[/Screen]" in content:
                start = content.find("[Screen]")
                end = content.find("[/Screen]") + len("[/Screen]")
                content = content[:start] + prefix.strip() + "\n\n" + content[end:]
            else:
                content = prefix + content
            m["content"] = content
            return
    messages.insert(0, {"role": "system", "content": prefix})


def _inject_multimodal(context: BeforeChatContext, image_bytes: bytes, query: str) -> None:
    provider = _get_llm_provider()
    mime = detect_mime_type(image_bytes)
    if provider == "Claude":
        _inject_claude_style(context, image_bytes, mime, query)
    else:
        _inject_openai_style(context, image_bytes, mime, query, provider)


def _inject_openai_style(context: BeforeChatContext, image_bytes: bytes, mime: str, query: str, provider: str) -> None:
    url = data_uri(image_bytes, mime)
    image_block = {"type": "image_url", "image_url": {"url": url, "detail": "auto"}}
    _inject_content_block(context, image_block, query)


def _inject_claude_style(context: BeforeChatContext, image_bytes: bytes, mime: str, query: str) -> None:
    b64 = encode_base64(image_bytes)
    image_block = {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}}
    _inject_content_block(context, image_block, query)


def _inject_content_block(context: BeforeChatContext, image_block: dict, query: str) -> None:
    messages: List[dict] = context.messages
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            original = messages[i]
            text_content = original.get("content", "")
            blocks: list = []
            if text_content and isinstance(text_content, str) and text_content.strip():
                blocks.append({"type": "text", "text": text_content})
            elif query:
                blocks.append({"type": "text", "text": query})
            else:
                blocks.append({"type": "text", "text": "[Screenshot attached]"})
            blocks.append(image_block)
            original["content"] = blocks
            return


def _inject_description(context: BeforeChatContext, image_bytes: bytes, query: str, cfg: CloudVisionConfig) -> None:
    from plugins.cloud_vision.vision_provider import VisionProviderRegistry

    prompt = query or "Describe what is currently visible on the screen in detail."
    try:
        provider = VisionProviderRegistry.get(cfg.vision_provider)
        provider.api_key = cfg.vision_api_key
        provider.base_url = cfg.vision_base_url
        provider.model = cfg.vision_model
        mime = detect_mime_type(image_bytes)
        description = provider.describe_image(image_bytes, mime, prompt)
    except Exception as exc:
        logger.warning("Tool vision API failed in hook: %s", exc)
        return

    if not description:
        return

    prefix = f"[Screen Context]\n{description}\n[/Screen Context]\n\n"
    messages: List[dict] = context.messages
    for m in messages:
        if m.get("role") == "system":
            m["content"] = prefix + (m.get("content") or "")
            return
    messages.insert(0, {"role": "system", "content": prefix})
