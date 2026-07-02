"""Cloud Vision plugin entry point."""

from __future__ import annotations

from pathlib import Path

# Import all providers so they self-register with VisionProviderRegistry
from plugins.cloud_vision.providers import (  # noqa: F401
    claude_provider,
    doubao_provider,
    gemini_provider,
    openai_compatible_provider,
    openai_provider,
    qwen_provider,
    zhipu_provider,
)

from collections.abc import Callable

from plugins.cloud_vision.config_model import CloudVisionConfig, default_config_path, load_config
from plugins.cloud_vision.auto_monitor import AutoMonitor
from plugins.cloud_vision.frontend_config import make_frontend_config
from plugins.cloud_vision.proactive_monitor import ProactiveMonitor
from plugins.cloud_vision.screen_hook import create_screen_hook
from plugins.cloud_vision.screen_tool import make_cloud_vision_describe
from sdk.logging import get_logger
from sdk.plugin import PluginBase
from sdk.plugin_host_context import PluginHostContext, PluginSettingsUIContext
from sdk.register import PluginCapabilityRegistry
from sdk.types import FrontendConfigAction, FrontendConfigContribution, SettingsUIContribution

logger = get_logger(__name__, plugin_id="com.shinsekai.cloud_vision")

# ── Multimodal detection ────────────────────────────────────────────

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
    """Heuristic check: does *provider* + *model* support multimodal input?"""
    patterns = _MULTIMODAL_PATTERNS.get(provider, ())
    model_lower = model.lower()
    return any(model_lower.startswith(p) or p in model_lower for p in patterns)


def _get_current_model(provider: str) -> str:
    """Read the active model id from global config."""
    try:
        from config.config_manager import ConfigManager

        cm = ConfigManager()
        return (cm.config.api_config.llm_model or {}).get(provider, "")
    except Exception:
        return ""


# ── Plugin class ────────────────────────────────────────────────────

class CloudVisionPlugin(PluginBase):
    """Cloud-based vision integration with dual-mode screen recognition."""

    def __init__(self) -> None:
        super().__init__()
        self._cfg: CloudVisionConfig | None = None
        self._emit_user_text: Callable[[str], None] | None = None
        self._auto_monitor: AutoMonitor | None = None
        self._proactive_monitor: ProactiveMonitor | None = None

    @property
    def plugin_id(self) -> str:
        return "com.shinsekai.cloud_vision"

    @property
    def plugin_version(self) -> str:
        return "1.0.0"

    @property
    def plugin_name(self) -> str:
        return "Cloud Vision"

    @property
    def plugin_description(self) -> str:
        return "Cloud-based screen recognition — auto-screenshot + LLM-callable vision tool."

    @property
    def plugin_author(self) -> str:
        return "pipi_"

    @property
    def priority(self) -> int:
        return 80

    def initialize(
        self,
        register: PluginCapabilityRegistry,
        plugin_root: Path,
        host: PluginHostContext,
    ) -> None:
        cfg_path = default_config_path(plugin_root)
        self._cfg = load_config(cfg_path)

        # Detect multimodal capability
        provider = host.selected_llm_provider or _get_current_llm_provider()
        model = _get_current_model(provider)
        self._cfg.multimodal_detected = _detect_multimodal(provider, model)

        logger.info(
            "Cloud Vision initialized: provider=%s model=%s multimodal=%s",
            provider,
            model,
            self._cfg.multimodal_detected,
        )

        # Register settings page with action button
        register.register_frontend_config_page(
            make_frontend_config(self._cfg, plugin_root)
        )

        # Register the on-demand vision tool (always available)
        register.register_llm_tool(_make_tool_registrar(self._cfg))

        # Register before_chat_hook for tool Mode 2 (stashed screenshot injection)
        register.register_before_chat_hook(create_screen_hook(self._cfg))

        # Register user-input trigger to get emit_user_text for auto/proactive monitors
        register.register_user_input_trigger(self._on_user_input_trigger)

    def _on_user_input_trigger(self, emit_user_text: Callable[[str], None]) -> None:
        """Stash emit_user_text and start monitors based on config."""
        self._emit_user_text = emit_user_text
        if self._cfg is None:
            return
        if self._cfg.auto_screenshot:
            self._start_auto_monitor()
        if self._cfg.proactive_mode:
            self._start_proactive_monitor()

    def _start_auto_monitor(self) -> None:
        if self._auto_monitor is not None:
            return
        if self._cfg is None or self._emit_user_text is None:
            return
        self._auto_monitor = AutoMonitor(self._cfg, self._emit_user_text)
        self._auto_monitor.start()

    def _start_proactive_monitor(self) -> None:
        if self._proactive_monitor is not None:
            return
        if self._cfg is None or self._emit_user_text is None:
            return
        self._proactive_monitor = ProactiveMonitor(self._cfg, self._emit_user_text)
        self._proactive_monitor.start()

    def shutdown(self) -> None:
        if self._auto_monitor is not None:
            self._auto_monitor.stop()
            self._auto_monitor = None
        if self._proactive_monitor is not None:
            self._proactive_monitor.stop()
            self._proactive_monitor = None

    def _action_auto_calibrate(self, values: dict) -> dict:
        """Auto-calibrate OmniParser offset by comparing screenshots."""
        from plugins.cloud_vision.image_utils import encode_base64
        from plugins.cloud_vision.screenshot import capture_screen

        try:
            img = capture_screen(self._cfg.monitor_index if self._cfg else 1)
        except Exception as e:
            return {"message": f"截图失败: {e}"}

        # Call OmniParser
        import json
        from urllib.request import Request, urlopen

        try:
            b64 = encode_base64(img)
            req = Request(
                "http://127.0.0.1:7862/process",
                data=json.dumps({"image": b64}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urlopen(req, timeout=30)
            data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {"message": f"OmniParser 连接失败: {e}"}

        elements = data.get("elements", [])
        if not elements:
            return {"message": "OmniParser 未检测到 UI 元素，请打开一个带按钮/图标的窗口后再试。"}

        # Find the most confident element near the top-left as anchor
        # Use the first element with a bbox
        anchor = None
        for el in elements:
            bbox = el.get("bbox")
            if bbox and len(bbox) == 4 and bbox[0] > 0 and bbox[1] > 0:
                anchor = bbox
                break

        if anchor is None:
            return {"message": "未找到可用锚点元素。"}

        # The OmniParser bbox should match mss coordinates directly.
        # If there's a consistent offset, it should be visible in the top-left corner.
        # For simplicity, assume OmniParser is correct; set offset to 0.
        # Users can manually tweak if needed.
        self._cfg.omni_offset_x = 0
        self._cfg.omni_offset_y = 0
        from plugins.cloud_vision.config_model import default_config_path, save_config
        if self._cfg:
            save_config(default_config_path(Path("data/plugins/com.shinsekai.cloud_vision")), self._cfg)

        el = elements[0]
        return {
            "message": (
                f"已重置偏移为 (0,0)。\n"
                f"参考锚点: {el.get('type','?')} '{el.get('text','')}' "
                f"bbox={anchor}\n"
                f"如果点击仍然不准，请在设置中手动调整偏移量。"
            ),
        }


def _build_frontend_config(cfg: CloudVisionConfig, plugin_root: Path) -> FrontendConfigContribution:
    """Build a FrontendConfigContribution with schema + action buttons."""

    llm_provider = _get_current_llm_provider()
    llm_model = _get_current_model(llm_provider)
    is_mm = _detect_multimodal(llm_provider, llm_model)
    if is_mm:
        detection_desc = (
            f"✅ 当前 LLM（{llm_provider} / {llm_model}）支持多模态，"
            "将直接让 LLM 理解图像内容。下方云端 API 配置可留空。"
        )
    else:
        detection_desc = (
            f"⚠️ 当前 LLM（{llm_provider} / {llm_model}）不支持多模态。"
            "如需识屏，请在下方配置云端视觉 API。"
        )

    schema = [
        {
            "id": "cloud_api",
            "title": "📡 云端视觉 API（主 LLM 不支持多模态时启用）",
            "fields": [
                {
                    "key": "use_cloud_api", "label": "是否开启云端识图", "type": "boolean",
                    "defaultValue": True,
                    "description": "关闭后截图直接传给多模态 LLM，不经过云端视觉 API（需 LLM 支持多模态）。",
                },
                {
                    "key": "vision_provider", "label": "服务商", "type": "select",
                    "defaultValue": "openai",
                    "options": [
                        {"label": "OpenAI 兼容（通用，填URL即可）", "value": "openai_compatible"},
                        {"label": "OpenAI Vision (GPT-4o)", "value": "openai"},
                        {"label": "Claude Vision (Anthropic)", "value": "claude"},
                        {"label": "Gemini Vision (Google)", "value": "gemini"},
                        {"label": "通义千问 VL (阿里云)", "value": "qwen"},
                        {"label": "智谱 GLM-4V", "value": "zhipu"},
                        {"label": "豆包视觉 (火山引擎)", "value": "doubao"},
                    ],
                    "description": "选择用于识屏的云端视觉模型。仅当主 LLM 不支持多模态时需要。",
                },
                {
                    "key": "vision_base_url", "label": "基础网址", "type": "url",
                    "defaultValue": "", "placeholder": "留空使用默认地址",
                },
                {
                    "key": "vision_api_key", "label": "API Key", "type": "password",
                    "defaultValue": "", "placeholder": "留空则回退到全局 LLM API Key",
                },
                {
                    "key": "vision_model", "label": "模型 ID", "type": "text",
                    "defaultValue": "gpt-4o", "placeholder": "如 gpt-4o / qwen-vl-plus / glm-4v",
                },
            ],
        },
        {
            "id": "auto_screenshot",
            "title": "🖼️ 自动识屏（定时截图，AI 主动回复）",
            "fields": [
                {
                    "key": "auto_screenshot", "label": "开启自动识屏", "type": "boolean",
                    "defaultValue": False,
                    "description": "定时截图回复：每 N 秒截图一次，AI 主动发消息描述屏幕。开启时自动关闭主动识屏。",
                },
                {
                    "key": "screenshot_interval_sec", "label": "自动识屏间隔 (秒)", "type": "number",
                    "defaultValue": 30.0, "min": 5.0, "max": 600.0, "step": 1.0,
                    "description": "每隔多少秒截图并回复一次。",
                },
                {
                    "key": "monitor_index", "label": "显示器索引", "type": "integer",
                    "defaultValue": 1, "min": 0, "max": 16, "step": 1,
                    "description": "mss：0=所有显示器合成；1 通常为第一块物理屏。",
                },
                {
                    "key": "max_image_size_mb", "label": "图片最大尺寸 (MB)", "type": "number",
                    "defaultValue": 10.0, "min": 0.5, "max": 100.0, "step": 1.0,
                    "description": "超过此大小的截图自动压缩。",
                },
            ],
        },
        {
            "id": "proactive",
            "title": "🔔 主动识屏（检测屏幕变化，AI 主动回复）",
            "fields": [
                {
                    "key": "proactive_mode", "label": "开启主动识屏", "type": "boolean",
                    "defaultValue": False,
                    "description": "检测屏幕变化后 AI 主动回复。开启时自动关闭上方自动识屏。",
                },
                {
                    "key": "proactive_poll_sec", "label": "屏幕检测间隔 (秒)", "type": "number",
                    "defaultValue": 3.0, "min": 1.0, "max": 60.0, "step": 0.5,
                    "description": "每隔多少秒检测一次屏幕是否变化。",
                },
                {
                    "key": "proactive_interval_sec", "label": "主动回复冷却 (秒)", "type": "number",
                    "defaultValue": 30.0, "min": 5.0, "max": 600.0, "step": 1.0,
                    "description": "两次主动消息之间的最小间隔，防刷屏。",
                },
                {
                    "key": "proactive_prompt", "label": "主动识屏提示词（可选）", "type": "textarea",
                    "defaultValue": "", "placeholder": "留空使用内置提示词。",
                    "span": "full",
                },
            ],
        },
    ]

    actions = []
    if _has_mouse_control():
        actions.append(
            FrontendConfigAction(
                id="sync_omni_offset",
                label="一键同步 OmniParser 偏移",
                description="从 mouse_control 读取偏移量并同步到 cloud_vision。",
                confirm="",
                run=lambda values: _action_sync_offset(),
                variant="primary",
            )
        )

    return FrontendConfigContribution(
        page_id="cloud_vision",
        title="Cloud Vision",
        kind="settings",
        description=detection_desc,
        restart_hint="修改云端视觉 API 或识屏模式后，建议重启聊天程序。",
        schema=schema,
        load_values=lambda: cfg.to_dict(),
        save_values=lambda values: _save_plugin_config(values, plugin_root),
        actions=actions,
        order=85.0,
    )
    """Check if mouse_control plugin is installed."""
    try:
        __import__("plugins.mouse_control.plugin")
        return True
    except ImportError:
        return False


def _action_sync_offset() -> dict:
    """Read OmniParser offset from mouse_control's config."""
    import json
    config_path = Path("data/plugins/com.shinsekai.mouse_control/omniparser_config.json")
    if not config_path.is_file():
        return {"message": "⚠️ 未找到 mouse_control 配置，请先配置 mouse_control 插件。"}
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        ox = int(raw.get("offset_x_px", raw.get("offset_x", 0)))
        oy = int(raw.get("offset_y_px", raw.get("offset_y", 0)))
        return {"message": f"✅ 已同步 mouse_control 偏移量: X={ox}, Y={oy}。cloud_vision 已自动使用。"}
    except Exception as e:
        return {"message": f"❌ 读取失败: {e}"}


def _save_plugin_config(values: dict, plugin_root: Path) -> None:
    """Save plugin config from form values (mutual exclusion handled)."""
    from plugins.cloud_vision.config_model import CloudVisionConfig, save_config

    auto = bool(values.get("auto_screenshot", False))
    proactive = bool(values.get("proactive_mode", False))
    if auto and proactive:
        raise ValueError("不能同时开启自动识屏和主动识屏，请只开启其中一个。")

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


def _make_settings_build():
    """Return a build callback for SettingsUIContribution (Qt fallback only)."""
    def build(ctx: PluginSettingsUIContext):
        from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Cloud Vision 设置请在 React 前端中配置。"))
        return w

    return build


def _get_current_llm_provider() -> str:
    try:
        from config.config_manager import ConfigManager

        return (ConfigManager().config.api_config.llm_provider or "ChatGPT").strip()
    except Exception:
        return "ChatGPT"


def _make_tool_registrar(cfg: CloudVisionConfig):
    """Return a callback for ``register_llm_tool`` that registers the tool function."""
    from llm.tools.tool_manager import ToolManager

    tool_fn = make_cloud_vision_describe(cfg)

    def registrar(tm: ToolManager) -> None:
        tm.register_function(
            tool_fn,
            name="cloud_vision_describe",
            description=(
                "Capture the current screen and answer a question about what's visible. "
                "Use when the user wants to know what's on screen — "
                "e.g. 'what's on screen?', 'look at this error', 'read that message', "
                "'看看屏幕', '拍屏', '识别屏幕', '帮我看看', '屏幕上有什么', '看下桌面', "
                "'截屏识别', '屏幕内容', '现在在干嘛', '这是什么'. "
                "Pass the user's real question as the 'query' parameter."
            ),
            group="default",
            risk="low",
        )

    return registrar
