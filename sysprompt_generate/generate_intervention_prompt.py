#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
智能体脚本，用于分析langrepo中的最新语音分析报告，并生成AI干预建议的prompt
生成的prompt将保存到langpmp文件夹中
"""

import os
import sys
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('sysprompt_generate.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file | 加载.env文件中的环境变量
from dotenv import load_dotenv
load_dotenv()

from sysprompt_generate.intervention_prompt_generator import InterventionPromptGenerator


def main():
    """主函数"""
    logger.info("开始生成干预建议prompt")
    try:
        # 从环境变量获取API密钥
        api_key = os.getenv('DASHSCOPE_API_KEY')
        logger.info("已获取API密钥: %s", "是" if api_key else "否")
        
        generator = InterventionPromptGenerator(api_key=api_key)
        saved_file, prompt_content = generator.process_latest_report()
        print(f"✅ 干预建议prompt已生成并保存至: {saved_file}")
        print("\n" + "="*50)
        print("生成的干预建议Prompt内容:")
        print("="*50)
        print(prompt_content)
        print("="*50)
        
        logger.info("干预建议生成成功，保存路径: %s", saved_file)
        return saved_file
    except Exception as e:
        print(f"❌ 生成干预建议prompt时出错: {e}")
        logger.error("生成干预建议时出错: %s", str(e), exc_info=True)
        return None


if __name__ == "__main__":
    main()