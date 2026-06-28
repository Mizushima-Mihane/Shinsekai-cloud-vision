"""LLM-callable ``cloud_vision_describe`` tool (Mode 1 & 2)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from plugins.cloud_vision.config_model import CloudVisionConfig
from plugins.cloud_vision.image_utils import compress_if_needed, detect_mime_type, encode_base64
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


# ── Optional OmniParser bridge ──────────────────────────────────────

def _get_omni_offsets(cfg: CloudVisionConfig) -> tuple[int, int]:
    """Get OmniParser coordinate offsets from mouse_control's config."""
    try:
        config_path = Path("data/plugins/com.shinsekai.mouse_control/omniparser_config.json")
        if config_path.is_file():
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            ox = int(raw.get("offset_x_px", raw.get("offset_x", 0)))
            oy = int(raw.get("offset_y_px", raw.get("offset_y", 0)))
            return ox, oy
    except Exception:
        pass
    return 0, 0


def _call_omniparser(image_bytes: bytes) -> dict[str, Any] | None:
    """Try to get structured UI elements from OmniParser if available."""
    try:
        from urllib.request import urlopen, Request
        urlopen("http://127.0.0.1:7862/health", timeout=1)
    except Exception:
        return None

    try:
        b64 = encode_base64(image_bytes)
        req = Request(
            "http://127.0.0.1:7862/process",
            data=json.dumps({"image": b64}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urlopen(req, timeout=15)
        return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.debug("OmniParser bridge failed: %s", exc)
        return None


def _format_omniparser_elements(omni_result: dict[str, Any], cfg: CloudVisionConfig) -> str:
    """Convert OmniParser output to concise text with coordinates,
    applying omni_offset from config."""
    elements = omni_result.get("elements", [])
    if not elements:
        return ""

    off_x, off_y = _get_omni_offsets(cfg)
    lines = ["\n[UI Elements — OmniParser detected:]"]
    if off_x or off_y:
        lines.append(f"  (offset: x+{off_x}, y+{off_y})")

    for el in elements[:30]:
        el_type = el.get("type", "?")
        text = (el.get("text") or "").strip()
        bbox = el.get("bbox", [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = [v + off_x if i % 2 == 0 else v + off_y for i, v in enumerate(bbox)]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            coord = f"center=({cx},{cy}) bbox=[{x1},{y1},{x2},{y2}]"
        else:
            coord = "no bbox"
        label = f"  [{el_type}] {text}" if text else f"  [{el_type}]"
        lines.append(f"{label} {coord}")
    return "\n".join(lines)


def make_cloud_vision_describe(
    cfg: CloudVisionConfig,
) -> Callable[..., Any]:
    """Build the tool function, capturing *cfg* by closure."""

    def cloud_vision_describe(query: str) -> str:
        """Capture the current screen and answer a question about it.

        Call this tool when the user wants to know what's visible on screen.
        If OmniParser is running, precise UI element coordinates are included.
        Use coordinates from OmniParser for accurate clicking with mouse_omniparser_click.

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

        result_parts: list[str] = []

        if cfg.multimodal_detected:
            stash_screenshot(img_bytes, query)
            result_parts.append("[Screenshot captured — image attached to the conversation context]")
        else:
            try:
                provider = VisionProviderRegistry.get(cfg.vision_provider)
                provider.api_key = cfg.vision_api_key
                provider.base_url = cfg.vision_base_url
                provider.model = cfg.vision_model
                mime = detect_mime_type(img_bytes)
                result = provider.describe_image(img_bytes, mime, query)
                result_parts.append(result)
            except Exception as exc:
                logger.debug("Vision API call failed in tool: %s", exc)
                result_parts.append(f"[Cloud Vision] Vision API error: {exc}")

        # Try OmniParser for structured element data
        omni = _call_omniparser(img_bytes)
        if omni:
            result_parts.append(_format_omniparser_elements(omni, cfg))

        return "\n".join(result_parts) if result_parts else "[Cloud Vision] No result"

    cloud_vision_describe.__name__ = "cloud_vision_describe"
    return cloud_vision_describe
