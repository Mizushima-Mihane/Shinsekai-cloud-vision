"""FrontendConfigContribution builder for Cloud Vision settings page."""

from __future__ import annotations

from pathlib import Path

from plugins.cloud_vision.config_model import CloudVisionConfig, default_config_path, save_config
from sdk.types import FrontendConfigContribution

# Multimodal detection patterns (duplicated to avoid circular import)
_MULTIMODAL_PATTERNS: dict[str, tuple[str, ...]] = {
    "ChatGPT": ("gpt-4o", "gpt-4-vision", "gpt-4-turbo", "gpt-4.1"),
    "Claude": ("claude-3-5", "claude-3-opus", "claude-3-sonnet", "claude-4"),
    "Gemini": ("gemini-2.0", "gemini-2.5", "gemini-1.5-pro", "gemini-1.5-flash"),
    "通义千问": ("qwen-vl", "qvq"),
    "智谱AI": ("glm-4v",),
    "豆包": ("doubao-vision",),
    "Deepseek": (),
    "Ollama": ("llava", "bakllava", "minicpm-v", "cogvlm"),
}


def _detect_multimodal(provider: str, model: str) -> bool:
    patterns = _MULTIMODAL_PATTERNS.get(provider, ())
    model_lower = model.lower()
    return any(model_lower.startswith(p) or p in model_lower for p in patterns)


def _get_current_model(provider: str) -> str:
    try:
        from config.config_manager import ConfigManager
        return (ConfigManager().config.api_config.llm_model or {}).get(provider, "")
    except Exception:
        return ""


def _get_current_llm_provider() -> str:
    try:
        from config.config_manager import ConfigManager
        return (ConfigManager().config.api_config.llm_provider or "ChatGPT").strip()
    except Exception:
        return "ChatGPT"


# Schema builder
def _f(key: str, label: str, ftype: str, *,
       default=None, desc="", placeholder="",
       options=None, min_v=None, max_v=None, step=None, span=None) -> dict:
    f: dict = {"key": key, "label": label, "type": ftype}
    if default is not None:
        f["defaultValue"] = default
    if desc:
        f["description"] = desc
    if placeholder:
        f["placeholder"] = placeholder
    if options:
        f["options"] = [{"label": lbl, "value": val} for lbl, val in options]
    if min_v is not None:
        f["min"] = min_v
    if max_v is not None:
        f["max"] = max_v
    if step is not None:
        f["step"] = step
    if span:
        f["span"] = span
    return f


# Config save helper
def _save(values: dict, plugin_root: Path) -> None:
    auto = bool(values.get("auto_screenshot", False))
    proactive = bool(values.get("proactive_mode", False))
    if auto and proactive:
        raise ValueError("Cannot enable both auto and proactive screenshot.")
    cfg = CloudVisionConfig(
        use_cloud_api=bool(values.get("use_cloud_api", True)),
        vision_provider=str(values.get("vision_provider") or "openai").strip().lower(),
        vision_base_url=str(values.get("vision_base_url") or "").strip(),
        vision_api_key=str(values.get("vision_api_key") or "").strip(),
        vision_model=str(values.get("vision_model") or "gpt-4o").strip(),
        auto_screenshot=auto,
        screenshot_interval_sec=float(values.get("screenshot_interval_sec", 30)),
        monitor_index=int(values.get("monitor_index", 1)),
        max_image_size_mb=float(values.get("max_image_size_mb", 10)),
        proactive_mode=proactive,
        proactive_poll_sec=float(values.get("proactive_poll_sec", 3)),
        proactive_interval_sec=float(values.get("proactive_interval_sec", 30)),
        proactive_prompt=str(values.get("proactive_prompt") or "").strip(),
    )
    save_config(default_config_path(plugin_root), cfg)


# Public API
def make_frontend_config(cfg: CloudVisionConfig, plugin_root: Path) -> FrontendConfigContribution:
    llm_provider = _get_current_llm_provider()
    llm_model = _get_current_model(llm_provider)
    is_mm = _detect_multimodal(llm_provider, llm_model)
    desc = (
        f"当前 LLM（{llm_provider} / {llm_model}）支持多模态，可直接理解图像内容。"
        if is_mm else
        f"当前 LLM（{llm_provider} / {llm_model}）不支持多模态，需配置云端视觉 API。"
    )

    return FrontendConfigContribution(
        page_id="cloud_vision",
        title="Cloud Vision",
        kind="settings",
        description=desc,
        restart_hint="修改云端视觉 API 或识屏模式后，建议重启。",
        schema=_SCHEMA,
        load_values=lambda: cfg.to_dict(),
        save_values=lambda values: _save(values, plugin_root),
        actions=[],
        order=85.0,
    )


# Schema
_SCHEMA = [
    {
        "id": "cloud_api",
        "title": "云端视觉 API（主 LLM 不支持多模态时启用）",
        "fields": [
            _f("use_cloud_api", "是否开启云端识图", "boolean", default=True,
               desc="关闭后截图直接传给多模态 LLM（需 LLM 支持多模态）。"),
            _f("vision_provider", "服务商", "select", default="openai",
               options=[
                   ("OpenAI 兼容（通用）", "openai_compatible"),
                   ("OpenAI Vision (GPT-4o)", "openai"),
                   ("Claude Vision (Anthropic)", "claude"),
                   ("Gemini Vision (Google)", "gemini"),
                   ("通义千问 VL (阿里云)", "qwen"),
                   ("智谱 GLM-4V", "zhipu"),
                   ("豆包视觉 (火山引擎)", "doubao"),
               ]),
            _f("vision_base_url", "基础网址", "url", placeholder="留空使用默认地址"),
            _f("vision_api_key", "API Key", "password", placeholder="留空则回退到全局 LLM API Key"),
            _f("vision_model", "模型 ID", "text", default="gpt-4o",
               placeholder="如 gpt-4o / qwen-vl-plus / glm-4v"),
        ],
    },
    {
        "id": "auto_screenshot",
        "title": "自动识屏（定时截图，AI 主动回复）",
        "fields": [
            _f("auto_screenshot", "开启自动识屏", "boolean",
               desc="定时截图回复：每 N 秒截图一次，AI 主动发消息描述屏幕。"),
            _f("screenshot_interval_sec", "自动识屏间隔 (秒)", "number",
               default=30.0, min_v=5.0, max_v=600.0, step=1.0,
               desc="每隔多少秒截图并回复一次。"),
            _f("monitor_index", "显示器索引", "integer",
               default=1, min_v=0, max_v=16, step=1,
               desc="mss：0=所有显示器合成；1 通常为第一块物理屏。"),
            _f("max_image_size_mb", "图片最大尺寸 (MB)", "number",
               default=10.0, min_v=0.5, max_v=100.0, step=1.0,
               desc="超过此大小的截图自动压缩。"),
        ],
    },
    {
        "id": "proactive",
        "title": "主动识屏（检测屏幕变化，AI 主动回复）",
        "fields": [
            _f("proactive_mode", "开启主动识屏", "boolean",
               desc="检测屏幕变化后 AI 主动回复。开启时自动关闭上方自动识屏。"),
            _f("proactive_poll_sec", "屏幕检测间隔 (秒)", "number",
               default=3.0, min_v=1.0, max_v=60.0, step=0.5,
               desc="每隔多少秒检测一次屏幕是否变化。"),
            _f("proactive_interval_sec", "主动回复冷却 (秒)", "number",
               default=30.0, min_v=5.0, max_v=600.0, step=1.0,
               desc="两次主动消息之间的最小间隔，防刷屏。"),
            _f("proactive_prompt", "主动识屏提示词（可选）", "textarea",
               placeholder="留空使用内置提示词。", span="full"),
        ],
    },
]
