# Project Structure Description | 项目结构说明

## Directory Structure | 目录结构

```
omni-convsystem/
├── app.py                 # Enhanced main application (recommended) | 增强版主应用程序（推荐使用）
├── main.py                # Basic version main application | 基础版本主应用程序
├── config.py              # Configuration management module | 配置管理模块
├── .env                   # Environment variable configuration file | 环境变量配置文件
├── .env.example           # Environment variable configuration example file | 环境变量配置示例文件
├── requirements.txt       # Project dependency list | 项目依赖列表
└── README.md             # Project documentation | 项目说明文档
```

## File Descriptions | 文件说明

### app.py
This is the main application file that provides complete Qwen-Omni real-time conversation functionality:
这是项目的主应用程序文件，提供了完整的Qwen-Omni实时对话功能：
- Real-time speech recognition and synthesis | 实时语音识别与合成
- Text chat display | 文本聊天显示
- Audio playback control | 音频播放控制
- Graceful program exit mechanism | 优雅的程序退出机制
- Better error handling | 更好的错误处理

### main.py
Basic version of the main application, basically consistent with your original code.
基础版本的主应用程序，与您提供的原始代码基本一致。

### config.py
Configuration management module responsible for loading and managing environment variables:
配置管理模块，负责加载和管理环境变量：
- API key configuration | API密钥配置
- Region configuration (Singapore/Beijing) | 地域配置（新加坡/北京）
- Voice settings | 语音设置
- WebSocket URL configuration | WebSocket URL配置

### .env and .env.example
Environment variable configuration files:
环境变量配置文件：
- `.env.example` is an example configuration file | `.env.example` 是示例配置文件
- `.env` is the actual configuration file to be used (requires modification of API Key) | `.env` 是实际使用的配置文件（需要修改其中的API Key）

### requirements.txt
Project dependency list containing all required Python packages.
项目依赖列表，包含所有必需的Python包。

## Usage Instructions | 使用说明

1. Copy `.env.example` to `.env` and fill in your API Key:
   复制 `.env.example` 到 `.env` 并填入你的API Key:
   ```
   cp .env.example .env
   ```

2. Install dependencies:
   安装依赖:
   ```
   pip install -r requirements.txt
   ```

3. Run the program:
   运行程序:
   ```
   python app.py
   ```

4. Press `Ctrl+C` to stop the program.
   按 `Ctrl+C` 停止程序。