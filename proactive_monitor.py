"""Proactive screen monitor: detect changes and trigger AI responses."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

from plugins.cloud_vision.config_model import CloudVisionConfig
from plugins.cloud_vision.image_utils import compress_if_needed, detect_mime_type
from plugins.cloud_vision.screenshot import capture_screen
from plugins.cloud_vision.vision_provider import VisionProviderRegistry
from sdk.logging import get_logger

logger = get_logger(__name__, plugin_id="com.shinsekai.cloud_vision")

_DEFAULT_PROMPT = "Describe what just changed on the screen. If nothing significant changed, reply with a single period '.'"


class ProactiveMonitor:
    """Background thread that polls the screen and submits descriptions when
    the screen content changes significantly."""

    def __init__(
        self,
        cfg: CloudVisionConfig,
        emit_text: Callable[[str], None],
    ) -> None:
        self._cfg = cfg
        self._emit = emit_text
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_thumb: Optional[bytes] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="cloud-vision-monitor")
        self._thread.start()
        logger.info("Proactive monitor started: poll=%.1fs interval=%.1fs",
                    self._cfg.proactive_poll_sec, self._cfg.proactive_interval_sec)

    def stop(self) -> None:
        self._running = False
        logger.info("Proactive monitor stopped")

    @property
    def running(self) -> bool:
        return self._running

    # ── internal ─────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running:
            try:
                time.sleep(self._cfg.proactive_poll_sec)
                if not self._running:
                    break
                self._poll_once()
            except Exception as exc:
                logger.warning("Proactive monitor poll error: %s", exc)

    def _poll_once(self) -> None:
        # Cooldown
        if time.time() - self._cfg.last_proactive_time < self._cfg.proactive_interval_sec:
            return

        try:
            img = capture_screen(self._cfg.monitor_index)
        except Exception as exc:
            logger.warning("Proactive screenshot failed: %s", exc)
            return

        # Make thumbnail for diff comparison (64px wide)
        thumb = _make_thumbnail(img)
        if self._last_thumb is not None:
            diff = _pixel_diff(self._last_thumb, thumb)
            if diff < 0.05:  # less than 5% change, skip
                return

        self._last_thumb = thumb

        # Significant change detected
        img = compress_if_needed(img, self._cfg.max_image_size_mb)
        prompt = self._cfg.proactive_prompt or _DEFAULT_PROMPT

        if self._cfg.use_cloud_api:
            try:
                provider = VisionProviderRegistry.get(self._cfg.vision_provider)
                provider.api_key = self._cfg.vision_api_key
                provider.base_url = self._cfg.vision_base_url
                provider.model = self._cfg.vision_model
                mime = detect_mime_type(img)
                description = provider.describe_image(img, mime, prompt)
            except Exception as exc:
                logger.warning("Proactive vision API failed: %s", exc)
                return
            if not description or description.strip() in ("。", ".", ""):
                return
            self._cfg.last_proactive_time = time.time()
            from plugins.cloud_vision.screen_tool import stash_description
            stash_description(description.strip()[:800])
        else:
            from plugins.cloud_vision.screen_tool import stash_screenshot
            stash_screenshot(img, prompt)

        self._cfg.last_proactive_time = time.time()
        self._emit("[Screen]")
        logger.info("Proactive message sent: %d chars", len(description))


# ── Thumbnail / diff helpers ────────────────────────────────────────

def _make_thumbnail(image_bytes: bytes, width: int = 64) -> bytes:
    """Return a tiny PNG thumbnail for diff comparison."""
    try:
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("L")  # grayscale
        h = int(img.height * width / img.width)
        img = img.resize((width, h), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return image_bytes


def _pixel_diff(thumb_a: bytes, thumb_b: bytes) -> float:
    """Return fraction of pixels that differ between two thumbnail byte strings."""
    try:
        from PIL import Image
        import io

        a = Image.open(io.BytesIO(thumb_a))
        b = Image.open(io.BytesIO(thumb_b))
        if a.size != b.size:
            return 1.0
        import numpy as np

        arr_a = np.array(a, dtype=np.int16)
        arr_b = np.array(b, dtype=np.int16)
        diff_pixels = np.count_nonzero(np.abs(arr_a - arr_b) > 20)
        total = arr_a.size
        return diff_pixels / total if total > 0 else 0.0
    except Exception:
        return 1.0  # treat diff failure as changed
