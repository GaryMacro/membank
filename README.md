# Qwen-Omni Real-time Conversation System (V2.1)
# Qwen-Omni实时对话系统 (V2.1)

This project implements a real-time conversation system based on Qwen-Omni, supporting multi-region API access and voice activity detection (VAD). The system enables natural language interaction through real-time voice input and text-to-speech output.
本项目基于Qwen-Omni实现了一个实时对话系统，支持多地域API接入和语音活动检测（VAD）。系统通过实时语音输入和文本转语音输出实现自然语言交互。

## Features
## 功能特性

- Real-time voice conversation: Real-time speech capture and processing through microphone
- 文本转语音输出：将模型返回的文本结果合成为语音播放
- Voice activity detection (VAD): Automatic detection of user speech to avoid invalid recordings
- 多区域API支持：可配置使用北京或新加坡地域的DashScope API
- **Voice wake-up and exit: Say wake word to activate system and exit word to deactivate**
- **语音唤醒和退出：说出唤醒词激活系统，说出退出词停用系统**

## Technical Architecture
## 技术架构

- Main control loop pattern: main.py starts the main process and coordinates audio input, model calling, and speech output
- 配置中心化：通过config.py和.env（或环境变量）统一管理配置项
- Layered modular design:
  - app.py: Application logic entry or core service encapsulation
  - conversation_history.py: Responsible for maintaining conversation context
  - system_prompt.py + system_prompt_template.json: Managing initial prompt templates
  - mem_bank.py: Implementing long-term memory storage function (speculation)

## Environment Requirements
## 环境要求

- Python 3.7+
- DashScope SDK >=1.23.9
- PyAudio
- python-dotenv

## Quick Start
## 快速开始

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure API Key:
   Set `DASHSCOPE_API_KEY` in `.env` file or environment variables.
   在`.env`文件或环境变量中设置`DASHSCOPE_API_KEY`。

3. Run the system:
   ```bash
   python main.py
   ```

## Configuration
## 配置说明

All configurations are managed through the `.env` file:
所有配置均通过`.env`文件管理：

- `DASHSCOPE_API_KEY`: DashScope API Key
- `REGION`: API region, options are `beijing` or `singapore`
- `DEFAULT_VOICE`: Voice selection, options are `Jennifer`, `Katerina`, `Kiki`, `Ryan`, `Cherry`
- `WAKE_WORDS`: Voice wake-up words, multiple words separated by commas, default is `你好灵灵,灵灵你好,灵灵灵灵,云云,你好,在吗`
- `EXIT_WORDS`: Voice exit words, multiple words separated by commas, default is `再见灵灵,灵灵再见,退出对话,结束对话,再见,拜拜,退出`

## Usage
## 使用说明

1. Run the system:
   运行系统：
   ```bash
   python main.py
   ```

2. The system will automatically start and wait for the wake word.
   系统会自动启动并等待唤醒词。

3. Say any wake word (default: "你好灵灵", "灵灵你好" etc.) to activate the system.
   说出任意唤醒词（默认："你好灵灵"、"灵灵你好"等）激活系统。

4. Have a conversation with the system.
   与系统进行对话。

5. Say any exit word (default: "再见灵灵", "退出对话" etc.) to deactivate the system.
   说出任意退出词（默认："再见灵灵"、"退出对话"等）停用系统。

6. To reactivate, say any wake word again.
   要重新激活，再次说出任意唤醒词即可。

## Notes
## 注意事项

- API Key must be configured through environment variables, do not hardcode in the code.
- 建议在生产环境中使用更严格的密钥管理系统