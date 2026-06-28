"""Image encoding, format detection, and size compression utilities."""

from __future__ import annotations

import base64
import io
from typing import Tuple

from sdk.logging import get_logger

logger = get_logger(__name__, plugin_id="com.shinsekai.cloud_vision")

# ── MIME detection from file headers ──

_SIGNATURES: list[Tuple[bytes, str]] = [
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"RIFF", "image/webp"),  # full check: RIFF....WEBP
    (b"BM", "image/bmp"),
]


def detect_mime_type(image_bytes: bytes) -> str:
    """Inspect leading bytes and return a MIME type string (default ``"image/png"``)."""
    for sig, mime in _SIGNATURES:
        if image_bytes[: len(sig)] == sig:
            if mime == "image/webp":
                # RIFF container — verify WEBP subtype
                if len(image_bytes) >= 12 and image_bytes[8:12] == b"WEBP":
                    return "image/webp"
                continue
            return mime
    return "image/png"


def encode_base64(image_bytes: bytes) -> str:
    """Return base64-encoded string (no data-URI prefix)."""
    return base64.b64encode(image_bytes).decode("ascii")


def data_uri(image_bytes: bytes, mime_type: str) -> str:
    """Return a ``data:...;base64,...`` URI string."""
    b64 = encode_base64(image_bytes)
    return f"data:{mime_type};base64,{b64}"


def compress_if_needed(image_bytes: bytes, max_size_mb: float) -> bytes:
    """Re-encode image as JPEG if it exceeds *max_size_mb* (PIL required).

    Returns the original bytes when PIL is unavailable or compression is
    unnecessary.  No API keys or user data appear in log output.
    """
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb <= max_size_mb:
        return image_bytes

    try:
        from PIL import Image
    except ImportError:
        logger.debug("PIL not installed; skipping image compression")
        return image_bytes

    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception:
        logger.debug("Cannot open image for compression — passing through original bytes")
        return image_bytes

    # Scale down longest dimension to 2048 px if larger
    w, h = img.size
    longest = max(w, h)
    if longest > 2048:
        scale = 2048 / longest
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Re-encode as JPEG
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=80, optimize=True)
    compressed = buf.getvalue()

    logger.debug(
        "Image compressed: %d → %d bytes (threshold %.1f MB)",
        len(image_bytes),
        len(compressed),
        max_size_mb,
    )
    return compressed
