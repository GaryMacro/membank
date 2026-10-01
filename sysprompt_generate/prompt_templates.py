#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prompt模板处理模块，用于加载和管理干预建议生成的模板
"""

import os
import sys
import json
import logging
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置日志
logger = logging.getLogger(__name__)

# Load environment variables from .env file | 加载.env文件中的环境变量
from dotenv import load_dotenv
load_dotenv()


def load_prompt_templates() -> Dict[str, str]:
    """
    加载prompt模板
    
    Returns:
        包含所有prompt模板的字典
    """
    logger.info("加载prompt模板文件")
    template_path = os.path.join(os.path.dirname(__file__), 'intervention_prompt_template.json')
    with open(template_path, 'r', encoding='utf-8') as f:
        templates = json.load(f)
        logger.info("成功加载模板，键数量: %d", len(templates))
        return templates


def format_qwen_prompt(report: Dict[Any, Any], templates: Dict[str, str]) -> str:
    """
    格式化Qwen模型使用的prompt
    
    Args:
        report: 语音分析报告
        templates: 模板字典
        
    Returns:
        格式化后的prompt
    """
    logger.info("开始格式化Qwen提示词")
    
    # 提取关键分析数据
    pronunciation = report.get("pronunciation", {})
    content_analysis = report.get("content_analysis", {}).get("analysis", "")
    
    # 获取声学特征
    duration = pronunciation.get("duration", "N/A")
    pitch_mean = pronunciation.get("pitch", {}).get("mean", "N/A")
    intensity_mean = pronunciation.get("intensity", {}).get("mean", "N/A")
    jitter = pronunciation.get("jitter_shimmer", {}).get("jitter", "N/A")
    shimmer = pronunciation.get("jitter_shimmer", {}).get("shimmer", "N/A")
    vowel_comment = pronunciation.get("vowels", {}).get("comment", [])
    consonant_comment = pronunciation.get("consonants", {}).get("comment", [])
    
    # 格式化元音分析注释
    formatted_vowel_comment = ', '.join(vowel_comment) if isinstance(vowel_comment, list) else vowel_comment
    formatted_consonant_comment = ', '.join(consonant_comment) if isinstance(consonant_comment, list) else consonant_comment
    
    # 使用模板构建输入给Qwen-Plus的提示
    qwen_prompt = templates["qwen_prompt_template"].format(
        duration=duration,
        pitch_mean=pitch_mean,
        intensity_mean=intensity_mean,
        jitter=jitter,
        shimmer=shimmer,
        vowel_comment=formatted_vowel_comment,
        consonant_comment=formatted_consonant_comment,
        content_analysis=content_analysis
    )
    
    logger.info("Qwen提示词格式化完成，长度: %d 字符", len(qwen_prompt))
    
    # 打印格式化后的提示词
    logger.info("="*50)
    logger.info("Qwen提示词内容:")
    logger.info("="*50)
    logger.info(qwen_prompt)
    logger.info("="*50)
    
    return qwen_prompt