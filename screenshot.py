"""Cross-platform screen capture via ``mss``."""

from __future__ import annotations

from typing import List, Dict

from sdk.logging import get_logger

logger = get_logger(__name__, plugin_id="com.shinsekai.cloud_vision")


def available_monitors() -> List[Dict[str, int]]:
    """Return info dicts for every monitor known to mss.

    Each dict has keys: ``index``, ``width``, ``height``, ``left``, ``top``.
    Index 0 = virtual "all monitors" desktop; 1 = primary physical monitor.
    """
    try:
        import mss
    except ImportError:
        logger.warning("mss not installed — screen capture unavailable")
        return []

    with mss.mss() as sct:
        return [
            {
                "index": i,
                "width": m["width"],
                "height": m["height"],
                "left": m.get("left", 0),
                "top": m.get("top", 0),
            }
            for i, m in enumerate(sct.monitors)
        ]


def capture_screen(monitor_index: int = 1) -> bytes:
    """Capture a monitor and return PNG bytes.

    *monitor_index* follows mss convention: 0 = all monitors, 1 = primary.
    Falls back to index 0 if *monitor_index* is out of range.
    """
    try:
        import mss
    except ImportError as exc:
        raise RuntimeError(
            "mss is required for screen capture. Install with: pip install mss"
        ) from exc

    with mss.mss() as sct:
        monitors = sct.monitors
        idx = monitor_index if 0 <= monitor_index < len(monitors) else 0
        mon = monitors[idx]
        sct_img = sct.grab(mon)
        return mss.tools.to_png(sct_img.rgb, sct_img.size)
