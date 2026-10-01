#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
干预Prompt生成器类，分析语音分析报告并生成AI干预建议prompt
"""

import os
import sys
import json
import glob
import logging
from typing import Dict, Any

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

from sysprompt_generate.prompt_templates import load_prompt_templates, format_qwen_prompt


class InterventionPromptGenerator:
    """
    干预Prompt生成器，分析语音分析报告并生成AI干预建议prompt
    """
    
    def __init__(self, api_key: str = None):
        """初始化干预Prompt生成器"""
        logger.info("初始化干预Prompt生成器")
        
        # 确保langpmp目录存在
        if not os.path.exists("langpmp"):
            os.makedirs("langpmp")
            logger.info("创建langpmp目录")
            
        self.api_key = api_key or os.getenv('DASHSCOPE_API_KEY')
        logger.info("API密钥已设置: %s", "是" if self.api_key else "否")
        
        # 加载prompt模板
        self.prompt_templates = load_prompt_templates()
        logger.info("已加载prompt模板")
    
    def find_latest_report(self) -> str:
        """
        查找langrepo目录中最新的分析报告
        
        Returns:
            最新报告文件的路径
        """
        logger.info("查找langrepo目录中的最新分析报告")
        report_files = glob.glob("langrepo/*.json")
        if not report_files:
            logger.error("在langrepo目录中未找到报告文件")
            raise FileNotFoundError("No report files found in langrepo directory")
        
        # 根据文件创建时间找到最新的报告
        latest_report = max(report_files, key=os.path.getctime)
        logger.info("找到最新报告: %s", latest_report)
        return latest_report
    
    def load_report(self, report_path: str) -> Dict[Any, Any]:
        """
        加载分析报告
        
        Args:
            report_path: 报告文件路径
            
        Returns:
            解析后的报告数据
        """
        logger.info("加载报告文件: %s", report_path)
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
            logger.info("成功加载报告，分析时间: %s", report.get("analysis_time", "未知"))
            return report
    
    def generate_intervention_prompt_with_qwen(self, report: Dict[Any, Any]) -> str:
        """
        使用Qwen-Plus模型根据分析报告生成AI干预建议的prompt
        
        Args:
            report: 语音分析报告
            
        Returns:
            生成的干预建议prompt
        """
        logger.info("开始生成干预建议")
        
        # 使用模板构建输入给Qwen-Plus的提示
        qwen_prompt = format_qwen_prompt(report, self.prompt_templates)
        logger.info("已格式化Qwen提示词")

        # 导入DashScope并设置API密钥
        from dashscope import Generation
        import dashscope
        
        if self.api_key:
            dashscope.api_key = self.api_key
            logger.info("已设置DashScope API密钥")
        elif 'DASHSCOPE_API_KEY' in os.environ:
            dashscope.api_key = os.environ['DASHSCOPE_API_KEY']
            logger.info("从环境变量中获取DashScope API密钥")
        else:
            logger.error("未提供DashScope API密钥")
            raise ValueError("DashScope API key not provided and DASHSCOPE_API_KEY environment variable not set")
        
        logger.info("调用Qwen-Plus模型")
        logger.info("即将发送给Qwen-Plus的提示词长度: %d 字符", len(qwen_prompt))
        
        # 调用Qwen-Plus模型
        response = Generation.call(
            model="qwen-plus",
            prompt=qwen_prompt,
            extra_body={"enable_thinking": True},
            stream=False,
            result_format="message"
        )
        
        # 提取生成的内容
        if response.status_code == 200:
            content = response.output.choices[0].message.content
            logger.info("成功生成干预建议，响应长度: %d 字符", len(content))
            return content
        else:
            logger.error("Qwen-Plus API调用失败，状态码: %d，错误信息: %s", 
                        response.status_code, response.message)
            raise Exception(f"Qwen-Plus API call failed with status code {response.status_code}: {response.message}")
    
    def save_prompt(self, prompt: str, report_time: str) -> str:
        """
        保存生成的prompt到JSON文件到langpmp文件夹中
        
        Args:
            prompt: 生成的prompt文本
            report_time: 原始报告的时间戳
            
        Returns:
            保存的文件路径
        """
        # 生成文件名
        filename = f"langpmp/intervention_prompt_{report_time.replace(':', '-')}.json"
        logger.info("准备保存干预建议到文件: %s", filename)
        
        # 创建prompt数据
        prompt_data = {
            "source_report_time": report_time,
            "generated_time": self._get_current_time(),
            "intervention_prompt": prompt
        }
        
        # 保存到文件
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(prompt_data, f, ensure_ascii=False, indent=2)
        
        logger.info("成功保存干预建议到文件: %s", filename)
        return filename
    
    def _get_current_time(self) -> str:
        """
        获取当前时间的ISO格式字符串
        
        Returns:
            当前时间字符串
        """
        import datetime
        return datetime.datetime.now().isoformat()
    
    def process_latest_report(self) -> tuple:
        """
        处理最新的报告并生成干预建议prompt
        
        Returns:
            保存的prompt文件路径和生成的prompt内容
        """
        logger.info("开始处理最新报告")
        
        # 查找并加载最新报告
        report_path = self.find_latest_report()
        report = self.load_report(report_path)
        
        # 获取报告时间戳
        report_time = report.get("analysis_time", "unknown")
        logger.info("处理报告时间: %s", report_time)
        
        # 使用Qwen-Plus生成干预建议prompt
        prompt = self.generate_intervention_prompt_with_qwen(report)
        logger.info("干预建议生成完成")
        
        # 保存prompt到langpmp文件夹
        saved_file = self.save_prompt(prompt, report_time)
        logger.info("处理完成")
        
        return saved_file, prompt