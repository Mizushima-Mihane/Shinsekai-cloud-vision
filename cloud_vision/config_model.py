"""Cloud Vision plugin configuration model."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict


@dataclass
class CloudVisionConfig:
    """Plugin configuration persisted to ``plugin_root/config.json``."""

    # ── 云端视觉 API（主 LLM 不支持多模态时使用） ──
    use_cloud_api: bool = True               # 是否启用云端视觉 API
    vision_provider: str = "openai"
    vision_base_url: str = ""
    vision_api_key: str = ""
    vision_model: str = "gpt-4o"

    # ── 自动识屏（被动：你发消息时注入截图） ──
    auto_screenshot: bool = False
    screenshot_interval_sec: float = 30.0     # 注入截图的冷却间隔 (5–600)
    monitor_index: int = 1
    max_image_size_mb: float = 10.0

    # ── 主动识屏（屏幕变化时 AI 主动说话） ──
    proactive_mode: bool = False
    proactive_poll_sec: float = 3.0           # 检测间隔 (1–60)
    proactive_interval_sec: float = 30.0      # 主动回复冷却 (5–600)
    proactive_prompt: str = ""                # 自定义提示词

    # ── 运行时状态（不持久化） ──
    last_screenshot_time: float = field(default=0.0, compare=False, repr=False)
    last_proactive_time: float = field(default=0.0, compare=False, repr=False)
    multimodal_detected: bool = field(default=False, compare=False, repr=False)

    def clamp(self) -> None:
        self.screenshot_interval_sec = max(5.0, min(600.0, self.screenshot_interval_sec))
        self.proactive_poll_sec = max(1.0, min(60.0, self.proactive_poll_sec))
        self.proactive_interval_sec = max(5.0, min(600.0, self.proactive_interval_sec))
        self.monitor_index = max(0, min(16, self.monitor_index))
        self.max_image_size_mb = max(0.5, min(100.0, self.max_image_size_mb))
        self.vision_provider = (self.vision_provider or "openai").strip().lower()
        self.vision_model = (self.vision_model or "gpt-4o").strip()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("last_screenshot_time", None)
        d.pop("last_proactive_time", None)
        d.pop("multimodal_detected", None)
        return d


def default_config_path(plugin_root: Path) -> Path:
    return plugin_root / "config.json"


def load_config(path: Path) -> CloudVisionConfig:
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            cfg = CloudVisionConfig(
                use_cloud_api=bool(raw.get("use_cloud_api", True)),
                vision_provider=str(raw.get("vision_provider") or "openai"),
                vision_base_url=str(raw.get("vision_base_url") or ""),
                vision_api_key=str(raw.get("vision_api_key") or ""),
                vision_model=str(raw.get("vision_model") or "gpt-4o"),
                auto_screenshot=bool(raw.get("auto_screenshot", False)),
                screenshot_interval_sec=float(raw.get("screenshot_interval_sec", 30)),
                monitor_index=int(raw.get("monitor_index", 1)),
                max_image_size_mb=float(raw.get("max_image_size_mb", 10)),
                proactive_mode=bool(raw.get("proactive_mode", False)),
                proactive_poll_sec=float(raw.get("proactive_poll_sec", 3)),
                proactive_interval_sec=float(raw.get("proactive_interval_sec", 30)),
                proactive_prompt=str(raw.get("proactive_prompt") or ""),
            )
            cfg.clamp()
            return cfg
        except Exception:
            pass
    cfg = CloudVisionConfig()
    cfg.clamp()
    return cfg


def save_config(path: Path, cfg: CloudVisionConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg.clamp()
    path.write_text(
        json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
