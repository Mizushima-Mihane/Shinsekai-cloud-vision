"""LLM-callable ``cloud_vision_describe`` tool (Mode 1 & 2)."""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from plugins.cloud_vision.config_model import CloudVisionConfig
from plugins.cloud_vision.image_utils import compress_if_needed, detect_mime_type
from plugins.cloud_vision.screenshot import capture_screen
from plugins.cloud_vision.vision_provider import VisionProviderRegistry
from sdk.logging import get_logger

logger = get_logger(__name__, plugin_id="com.shinsekai.cloud_vision")

# Shared screenshot stash: tool and hook share the same captured image within one
# chat turn to avoid double capture when both paths trigger.
_stashed_image: Optional[bytes] = None
_stashed_query: str = ""

# Shared description stash: auto/proactive monitors stash screen descriptions here;
# the before_chat_hook picks them up and injects into system prompt.
_stashed_description: Optional[str] = None


def stash_description(desc: str) -> None:
    """Store a screen description for the before_chat_hook to inject."""
    global _stashed_description
    _stashed_description = desc


def pop_stashed_description() -> Optional[str]:
    """Retrieve and clear the stashed description."""
    global _stashed_description
    d = _stashed_description
    _stashed_description = None
    return d


def stash_screenshot(image_bytes: bytes, query: str = "") -> None:
    """Store a screenshot so the before_chat_hook can reuse it."""
    global _stashed_image, _stashed_query
    _stashed_image = image_bytes
    _stashed_query = query


def pop_stashed_screenshot() -> tuple[Optional[bytes], str]:
    """Retrieve and clear the stashed screenshot. Returns (image_bytes, query)."""
    global _stashed_image, _stashed_query
    img, q = _stashed_image, _stashed_query
    _stashed_image = None
    _stashed_query = ""
    return img, q


def make_cloud_vision_describe(
    cfg: CloudVisionConfig,
) -> Callable[..., Any]:
    """Build the tool function, capturing *cfg* by closure."""

    def cloud_vision_describe(query: str) -> str:
        """Capture the current screen and answer a question about it.

        Call this tool when the user wants to know what's visible on screen.

        Args:
            query: What to look for or describe.  Examples:
                   "Describe everything visible on screen",
                   "Find the login button",
                   "Read all form fields and their positions".
        """
        try:
            img_bytes = capture_screen(cfg.monitor_index)
        except Exception as exc:
            logger.warning("Screen capture failed in tool: %s", exc)
            return f"[Cloud Vision] Screen capture failed: {exc}"

        img_bytes = compress_if_needed(img_bytes, cfg.max_image_size_mb)
        cfg.last_screenshot_time = time.time()

        if cfg.multimodal_detected:
            stash_screenshot(img_bytes, query)
            return "[Screenshot captured — image attached to the conversation context]"

        try:
            provider = VisionProviderRegistry.get(cfg.vision_provider)
            provider.api_key = cfg.vision_api_key
            provider.base_url = cfg.vision_base_url
            provider.model = cfg.vision_model
            mime = detect_mime_type(img_bytes)
            return provider.describe_image(img_bytes, mime, query)
        except Exception as exc:
            logger.debug("Vision API call failed in tool: %s", exc)
            return f"[Cloud Vision] Vision API error: {exc}"

    cloud_vision_describe.__name__ = "cloud_vision_describe"
    return cloud_vision_describe
