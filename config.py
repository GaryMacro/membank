import os
from dotenv import load_dotenv

# Load environment variables from .env file | 加载.env文件中的环境变量
load_dotenv()

# DashScope API configuration | DashScope API配置
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', 'YOUR_DEFAULT_API_KEY_HERE')

# Region configuration - Optional values: 'singapore' or 'beijing' | 地域配置 - 可选值: 'singapore' 或 'beijing'
REGION = os.getenv('REGION', 'beijing')

# Voice settings | 语音设置
DEFAULT_VOICE = os.getenv('DEFAULT_VOICE', 'Jennifer')

# Wake words and exit words settings | 唤醒词和退出词设置
# Multiple wake words and exit words are supported, separated by commas
# 支持多个唤醒词和退出词，用逗号分隔
WAKE_WORDS = os.getenv('WAKE_WORDS', '你好灵灵,灵灵你好,灵灵灵灵,云云,Hello').split(',')
WAKE_WORDS = [word.strip() for word in WAKE_WORDS]  # Remove whitespace | 去除空格

EXIT_WORDS = os.getenv('EXIT_WORDS', '再见灵灵,灵灵再见,退出对话,结束对话,goodbye').split(',')
EXIT_WORDS = [word.strip() for word in EXIT_WORDS]  # Remove whitespace | 去除空格

# WebSocket URL configuration | WebSocket URL配置
WEBSOCKET_URLS = {
    'singapore': 'wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime',
    'beijing': 'wss://dashscope.aliyuncs.com/api-ws/v1/realtime'
}

# Audio format configuration | 音频格式配置
INPUT_AUDIO_FORMAT = 'PCM_16000HZ_MONO_16BIT'
OUTPUT_AUDIO_FORMAT = 'PCM_24000HZ_MONO_16BIT'