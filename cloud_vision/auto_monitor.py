"""Auto-screenshot monitor: timer-based screen capture and AI reply."""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from plugins.cloud_vision.config_model import CloudVisionConfig
from plugins.cloud_vision.image_utils import compress_if_needed, detect_mime_type
from plugins.cloud_vision.screenshot import capture_screen
from plugins.cloud_vision.vision_provider import VisionProviderRegistry
from sdk.logging import get_logger

logger = get_logger(__name__, plugin_id="com.shinsekai.cloud_vision")

_DEFAULT_AUTO_PROMPT = "Describe what is currently visible on the screen in detail."


class AutoMonitor:
    """Background thread: every *interval* seconds, take a screenshot and submit
    a description to the chat (timer-based, no change detection)."""

    def __init__(
        self,
        cfg: CloudVisionConfig,
        emit_text: Callable[[str], None],
    ) -> None:
        self._cfg = cfg
        self._emit = emit_text
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="cloud-vision-auto")
        self._thread.start()
        logger.info("Auto monitor started: interval=%.1fs", self._cfg.screenshot_interval_sec)

    def stop(self) -> None:
        self._running = False
        logger.info("Auto monitor stopped")

    @property
    def running(self) -> bool:
        return self._running

    def _loop(self) -> None:
        while self._running:
            try:
                time.sleep(self._cfg.screenshot_interval_sec)
                if not self._running:
                    break
                self._tick()
            except Exception as exc:
                logger.warning("Auto monitor error: %s", exc)

    def _tick(self) -> None:
        try:
            img = capture_screen(self._cfg.monitor_index)
        except Exception as exc:
            logger.warning("Auto screenshot failed: %s", exc)
            return

        img = compress_if_needed(img, self._cfg.max_image_size_mb)

        if self._cfg.use_cloud_api:
            try:
                provider = VisionProviderRegistry.get(self._cfg.vision_provider)
                provider.api_key = self._cfg.vision_api_key
                provider.base_url = self._cfg.vision_base_url
                provider.model = self._cfg.vision_model
                mime = detect_mime_type(img)
                description = provider.describe_image(img, mime, _DEFAULT_AUTO_PROMPT)
            except Exception as exc:
                logger.warning("Auto vision API failed: %s", exc)
                return
            if not description:
                return
            # Stash description for before_chat_hook, send minimal trigger
            from plugins.cloud_vision.screen_tool import stash_description
            stash_description(description.strip()[:800])
        else:
            from plugins.cloud_vision.screen_tool import stash_screenshot
            stash_screenshot(img, _DEFAULT_AUTO_PROMPT)

        self._emit("[Screen]")
        logger.info("Auto trigger sent")
