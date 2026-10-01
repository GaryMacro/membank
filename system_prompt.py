#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
System Prompt Management Module
系统提示词管理模块

This module handles the generation and management of system prompts
for the Qwen-Omni real-time conversation system. It integrates memory
content with predefined templates to create dynamic system prompts.
该模块处理Qwen-Omni实时对话系统的系统提示词生成和管理。
它将记忆内容与预定义模板结合，创建动态系统提示词。
"""

import os
import json
import glob
from pathlib import Path
from typing import Dict, Any, Optional
from copy import deepcopy

# Ensure the root directory is in the Python path
# 确保根目录在Python路径中
ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))


def load_system_prompt_template_txt() -> str:
    """
    Load system prompt template from txt file.
    从txt文件中加载系统提示词模板。
    
    Returns:
        str: System prompt template content | 系统提示词模板内容
    """
    # Define template path | 定义模板路径
    template_path = os.path.join(os.path.dirname(__file__), "system_prompt_template.txt")
    
    try:
        # Try to load template from file | 尝试从文件加载模板
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()
        return template
    except FileNotFoundError:
        # Fallback template if file not found | 如果文件未找到，使用备用模板
        print("System prompt template file not found, using default template | 系统提示词模板文件未找到，使用默认模板")
        return """You are a personal assistant named Xiao Yun. Please answer user questions accurately and kindly, always responding with a helpful attitude.
你是名为灵灵的个人助手。请准确且友好地回答用户问题，始终以乐于助人的态度回应。

## User Profile
## 用户画像
{user_profile}

## Previous Conversation Segments
## 历史对话片段
{conversation_segments}"""


def load_system_prompt_template_json() -> str:
    """
    Load system prompt template from JSON file.
    从JSON文件中加载系统提示词模板。
    
    Returns:
        str: System prompt template content | 系统提示词模板内容
    """
    # Define template path | 定义模板路径
    template_path = os.path.join(os.path.dirname(__file__), "system_prompt_template.json")
    
    try:
        # Try to load template from JSON file | 尝试从JSON文件加载模板
        with open(template_path, "r", encoding="utf-8") as f:
            template_data = json.load(f)
        return template_data.get("system_prompt", "")
    except FileNotFoundError:
        # Return empty string if file not found | 如果文件未找到，返回空字符串
        print("System prompt template JSON file not found | 系统提示词模板JSON文件未找到")
        return ""


def load_system_prompt_template() -> str:
    """
    Load system prompt template from file (JSON优先，其次是TXT).
    从文件中加载系统提示词模板（优先JSON，其次是TXT）。
    
    Returns:
        str: System prompt template content | 系统提示词模板内容
    """
    # Try to load from JSON first | 优先尝试从JSON加载
    json_template = load_system_prompt_template_json()
    if json_template:
        return json_template
    
    # Fallback to TXT template | 回退到TXT模板
    return load_system_prompt_template_txt()


def find_latest_intervention_prompt_file() -> Optional[str]:
    """
    Find the latest intervention prompt JSON file in the langpmp directory.
    查找langpmp目录中最新的干预提示JSON文件。
    
    Returns:
        str or None: Path to the latest intervention prompt file or None if not found
        最新干预提示文件的路径，如果未找到则返回None
    """
    try:
        # Define langpmp directory path | 定义langpmp目录路径
        langpmp_dir = os.path.join(os.path.dirname(__file__), "langpmp")
        if not os.path.exists(langpmp_dir):
            print("langpmp directory not found | 未找到langpmp目录")
            return None
            
        # Find all intervention prompt JSON files | 查找所有干预提示JSON文件
        prompt_files = glob.glob(os.path.join(langpmp_dir, "intervention_prompt_*.json"))
        if not prompt_files:
            print("No intervention prompt files found in langpmp directory | 在langpmp目录中未找到干预提示文件")
            return None
            
        # Find the latest file based on creation time | 根据创建时间找到最新的文件
        latest_file = max(prompt_files, key=os.path.getctime)
        return latest_file
    except Exception as e:
        print(f"Error finding latest intervention prompt file: {e} | 查找最新干预提示文件时出错: {e}")
        return None


def load_latest_intervention_prompt() -> Optional[Dict[str, Any]]:
    """
    Load the content of the latest intervention prompt JSON file.
    加载最新干预提示JSON文件的内容。
    
    Returns:
        dict or None: Content of the latest intervention prompt file or None if not found
        最新干预提示文件的内容，如果未找到则返回None
    """
    try:
        # Find the latest intervention prompt file | 查找最新的干预提示文件
        latest_file = find_latest_intervention_prompt_file()
        if not latest_file:
            return None
            
        # Load the content of the file | 加载文件内容
        with open(latest_file, "r", encoding="utf-8") as f:
            prompt_content = json.load(f)
        return prompt_content
    except Exception as e:
        print(f"Error loading latest intervention prompt: {e} | 加载最新干预提示时出错: {e}")
        return None


def load_memory_from_json(json_file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load memory content from JSON file.
    从JSON文件加载记忆内容。
    
    Args:
        json_file_path: Path to the memory JSON file | 记忆JSON文件的路径
        
    Returns:
        Optional[Dict[str, Any]]: Memory content dictionary or None if error | 记忆内容字典或错误时为None
    """
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            memory_content = json.load(f)
        return memory_content
    except FileNotFoundError:
        print(f"Memory file not found: {json_file_path} | 未找到记忆文件: {json_file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file: {e} | 解析JSON文件出错: {e}")
        return None


def format_user_profile(user_profile: Dict[str, Any]) -> str:
    """
    Format user profile into a readable string.
    将用户画像格式化为可读字符串。
    
    Args:
        user_profile: User profile dictionary | 用户画像字典
        
    Returns:
        str: Formatted user profile string | 格式化的用户画像字符串
    """
    if not user_profile:
        return "No user profile information available.\n暂无用户画像信息。"
    
    profile_lines = []
    for key, value in user_profile.items():
        if isinstance(value, list):
            profile_lines.append(f"{key}:")
            for item in value:
                profile_lines.append(f"  - {item}")
        else:
            profile_lines.append(f"{key}: {value}")
    
    return "\n".join(profile_lines)


def format_conversation_segments(conversation_segments: list) -> str:
    """
    Format conversation segments into a readable string.
    将对话片段格式化为可读字符串。
    
    Args:
        conversation_segments: List of conversation segments | 对话片段列表
        
    Returns:
        str: Formatted conversation segments string | 格式化的对话片段字符串
    """
    if not conversation_segments:
        return "No previous conversation segments available.\n暂无历史对话片段。"
    
    segment_lines = []
    for index, segment in enumerate(conversation_segments, start=1):
        if isinstance(segment, dict) and "summary" in segment:
            summary = segment["summary"]
            convr_num = segment.get("convr_num")
            convr_index = segment.get("convr_index")
            if convr_num and convr_index:
                tag = f"CONVR{convr_num}-{convr_index}"
                segment_lines.append(f"{tag}. {summary}")
            else:
                segment_lines.append(f"{index}. {summary}")
    
    return "\n".join(segment_lines) if segment_lines else "No previous conversation segments available.\n暂无历史对话片段。"


def generate_system_prompt_with_memory(memory_content: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate system prompt with integrated memory content.
    生成包含记忆内容的系统提示词
    
    Args:
        memory_content: Memory content with user profile and conversation segments
                       包含用户画像和对话片段的记忆内容
        
    Returns:
        str: Formatted system prompt with integrated memory
             包含整合记忆的格式化系统提示词
    """
    # Load system prompt template | 加载系统提示词模板
    template = load_system_prompt_template()
    
    # Extract memory components | 提取记忆组件
    if memory_content and isinstance(memory_content, dict):
        user_profile = memory_content.get("User Profile", {})
        conversation_segments = memory_content.get("Previous Conversation Segments", [])
    else:
        user_profile = {}
        conversation_segments = []
    
    # Format memory components | 格式化记忆组件
    user_profile_str = format_user_profile(user_profile)
    conversation_segments_str = format_conversation_segments(conversation_segments)
    
    # Load latest intervention prompt | 加载最新的干预提示
    intervention_prompt_content = "No specific intervention suggestions at this time. | 暂无特定干预建议。"
    try:
        intervention_prompt_data = load_latest_intervention_prompt()
        if intervention_prompt_data and "intervention_prompt" in intervention_prompt_data:
            intervention_prompt_content = intervention_prompt_data["intervention_prompt"]
    except Exception as e:
        print(f"Warning: Failed to load intervention prompt: {e} | 警告：加载干预提示失败: {e}")
    
    # Generate final system prompt by filling template with actual content
    # 通过将实际内容填入模板生成最终系统提示词
    try:
        system_prompt = template.format(
            user_profile=user_profile_str,
            conversation_segments=conversation_segments_str,
            intervention_suggestions=intervention_prompt_content
        )
    except KeyError as e:
        print(f"Missing key in system prompt template: {e} | 系统提示词模板中缺少键: {e}")
        # Use fallback template | 使用备用模板
        system_prompt = f"""You are a personal assistant named Xiao Yun. Please answer user questions accurately and kindly, always responding with a helpful attitude.
你是名为灵灵的个人助手。请准确且友好地回答用户问题，始终以乐于助人的态度回应。

## User Profile
## 用户画像
{user_profile_str}

## Previous Conversation Segments
## 历史对话片段
{conversation_segments_str}

## Language Intervention Suggestions
## 语言干预建议
{intervention_prompt_content}"""
    
    return system_prompt


def generate_system_prompt_from_files(template_file: str, memory_file: str) -> Optional[str]:
    """
    Generate system prompt by combining template and memory files.
    通过组合模板和记忆文件生成系统提示词。
    
    Args:
        template_file: Path to system prompt template file | 系统提示词模板文件路径
        memory_file: Path to memory JSON file | 记忆JSON文件路径
        
    Returns:
        Optional[str]: Generated system prompt or None if error | 生成的系统提示词，错误时为None
    """
    # Load memory content from JSON file | 从JSON文件加载记忆内容
    memory_content = load_memory_from_json(memory_file)
    if memory_content is None:
        return None
    
    # Save current template file path | 保存当前模板文件路径
    old_cwd = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    
    try:
        # Generate system prompt with memory content | 使用记忆内容生成系统提示词
        system_prompt = generate_system_prompt_with_memory(memory_content)
        return system_prompt
    finally:
        os.chdir(old_cwd)


def update_system_prompt_with_current_memory(conversation_history_manager) -> str:
    """
    Update system prompt with current memory from conversation history manager.
    使用对话历史管理器中的当前记忆更新系统提示词。
    
    Args:
        conversation_history_manager: Conversation history manager instance
                                     对话历史管理器实例
        
    Returns:
        str: Updated system prompt | 更新后的系统提示词
    """
    # Get current memory content | 获取当前记忆内容
    memory_content = conversation_history_manager.get_current_memory()
    
    # Generate system prompt with memory | 生成包含记忆的系统提示词
    return generate_system_prompt_with_memory(memory_content)


