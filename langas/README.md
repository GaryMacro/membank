# Language Analysis System (语言分析系统)

This module provides comprehensive audio analysis capabilities for language learning and assessment.

该模块为语言学习和评估提供全面的音频分析功能。

## Features (功能)

1. **Pronunciation Analysis (发音分析)**
   - Uses Praat-Parselmouth for phonetic analysis
   - Analyzes pitch, intensity, formants and other acoustic features
   - 使用Praat-Parselmouth进行语音分析
   - 分析音高、强度、共振峰等声学特征

2. **Content Analysis (内容分析)**
   - Uses Qwen-Audio model to analyze semantic, grammatical and lexical aspects
   - Provides comprehensive feedback on language usage
   - 使用Qwen-Audio模型分析语义、语法和词汇方面
   - 提供关于语言使用的全面反馈

3. **Report Generation (报告生成)**
   - Generates structured JSON reports
   - Saves reports to langrepo directory
   - 生成结构化JSON报告
   - 将报告保存到langrepo目录

## Installation (安装)

To use this module, install the required dependencies:

要使用此模块，请安装所需的依赖项：

```bash
pip install -r requirements.txt
```

Note: You may need to install Praat-Parselmouth separately depending on your system.

注意：您可能需要根据您的系统单独安装Praat-Parselmouth。

## Usage (使用方法)

Run the analyzer script to analyze the latest recording:

运行分析器脚本以分析最新的录音：

```bash
python langas/audio_analyzer.py
```

The system will:
1. Find the latest recording in the recordings directory
2. Analyze it using both Praat and Qwen-Audio
3. Generate a JSON report in the langrepo directory

系统将：
1. 在recordings目录中找到最新的录音
2. 使用Praat和Qwen-Audio进行分析
3. 在langrepo目录中生成JSON报告

## Output Format (输出格式)

The generated JSON report includes:
- Audio file information
- Pronunciation analysis data
- Content analysis from Qwen-Audio
- Timestamp of analysis

生成的JSON报告包括：
- 音频文件信息
- 发音分析数据
- Qwen-Audio的内容分析
- 分析时间戳

## Requirements (要求)

- Python 3.6+
- DashScope API key configured in .env file
- Praat-Parselmouth library
- Access to Qwen-Audio model through DashScope

- Python 3.6+
- 在.env文件中配置的DashScope API密钥
- Praat-Parselmouth库
- 通过DashScope访问Qwen-Audio模型