# Cloud Vision — Shinsekai Plugin

Cloud-based screen recognition plugin for [Shinsekai](https://github.com/RachelForster/Shinsekai).  
Let your AI see the screen — with auto/proactive screenshot, cloud vision API, and OmniParser integration.

## Features

| Mode | Description |
|------|-------------|
| **Tool-based** | LLM calls `cloud_vision_describe(query)` when user asks to look at the screen |
| **Auto Screenshot** | Timer: screenshot every N seconds, AI auto-replies |
| **Proactive Screen** | Change detection: AI only replies when screen content changes |
| **OmniParser Bridge** | If [mouse_control](https://github.com/pipi/mouse_control) is installed, attaches precise UI element coordinates to descriptions |

## Supported Vision Providers

| Provider | Default URL |
|----------|-------------|
| OpenAI GPT-4o | `https://api.openai.com/v1` |
| Claude Vision | Anthropic API |
| Gemini Vision | Google AI |
| Qwen VL (Alibaba) | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| GLM-4V (Zhipu) | `https://open.bigmodel.cn/api/paas/v4` |
| Doubao Vision (ByteDance) | `https://ark.cn-beijing.volces.com/api/v3` |
| OpenAI Compatible | Any OpenAI-compatible endpoint |

## Installation

1. Copy `cloud_vision/` into `<Shinsekai>/plugins/cloud_vision/`
2. Add to `<Shinsekai>/data/config/plugins.yaml`:
   ```yaml
   - entry: plugins.cloud_vision.plugin:CloudVisionPlugin
     enabled: true
   ```
3. Install dependency:
   ```bash
   pip install mss
   ```
4. Restart Shinsekai

## Configuration

Open **Settings → Cloud Vision** in the Shinsekai UI.

## Author

pipi_

## License

MIT
