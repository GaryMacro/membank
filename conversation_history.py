#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Conversation History and Memory Management
对话历史和记忆管理模块

This module handles the collection, storage, and processing of conversation history
for the Qwen-Omni real-time conversation system. It integrates with the langmem
memory processing system to extract user profiles and conversation segments.
该模块处理Qwen-Omni实时对话系统的对话历史收集、存储和处理。
它与langmem记忆处理系统集成，以提取用户档案和对话片段。
"""

import json
import os
import threading
import time
import importlib
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from copy import deepcopy
import asyncio
import concurrent.futures
import logging

# Configure logging
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("conversation_system.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Import the memory processing function
# 导入记忆处理函数
from mem_bank import (
    process_conversation_memory,
    incrementally_update_memory,
    load_memory_from_system_prompt_file,
)
from system_prompt import generate_system_prompt_with_memory, update_system_prompt_with_current_memory
from summarizer import generate_summary
from langmem.short_term import RunningSummary


class ConversationHistoryManager:
    """
    Manages conversation history and memory processing
    管理对话历史和记忆处理
    
    This class is responsible for collecting conversation messages,
    storing them, and processing them with the langmem system.
    这个类负责收集对话消息、存储它们，并使用langmem系统处理它们。
    """
    
    def __init__(
        self,
        history_file_prefix: str = "conversation_history",
        history_dir: str = "history"
    ):
        """
        Initialize the conversation history manager
        初始化对话历史管理器
        
        Args:
            history_file_prefix: Prefix for history files | 历史文件的前缀
            history_dir: Directory to store history files | 用于存储历史文件的目录
        """
        # Store the prefix for history files | 存储历史文件的前缀
        self.history_file_prefix = history_file_prefix

        # Store conversation messages | 存储对话消息
        self.messages: List[Dict[str, Any]] = []

        # Track running summary state for incremental compression | 跟踪增量压缩的运行摘要状态
        self._running_summary: Optional[RunningSummary] = None

        # Thread lock for thread-safe operations | 线程锁，用于线程安全操作
        self.lock = threading.Lock()

        # Track if memory processing is in progress | 跟踪记忆处理是否正在进行
        self.processing_memory = False

        # Store current memory content | 存储当前记忆内容
        self.current_memory = {
            "User Profile": {},
            "Previous Conversation Segments": []
        }

        # Buffer assistant deltas until the response finishes | 缓存助手的增量响应直到完整消息生成
        self._assistant_buffer: List[str] = []

        # Track last memory update time | 跟踪上次记忆更新时间
        self.last_memory_update = 0

        # Track whether session prompt needs refresh | 跟踪是否需要刷新会话提示词
        self.pending_prompt_refresh = False

        # Minimum interval between memory updates (seconds) | 记忆更新的最小间隔（秒）
        self.memory_update_interval = 30

        # Create a directory for history files if it doesn't exist | 如果不存在，创建历史文件目录
        self.history_dir = history_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.history_dir_path = Path(self.history_dir)
        self.history_dir_path.mkdir(parents=True, exist_ok=True)
        self.history_file_path = self.history_dir_path / f"{self.history_file_prefix}_{self.session_id}.json"
        self.transcript_file_path = self.history_dir_path / f"{self.history_file_prefix}_{self.session_id}_transcript.json"

        # Load existing memory system prompt if available | 加载现有记忆系统提示词
        self._load_memory_from_disk()
    
    def add_message(self, role: str, content: str):
        """
        Add a message to the conversation history
        添加消息到对话历史
        
        Args:
            role: Role of the message sender (user, assistant, system) | 消息发送者的角色（用户、助手、系统）
            content: Content of the message | 消息内容
        """
        logger.info(f"Adding message: role={role}, content_length={len(content)}")
        
        # Create message object | 创建消息对象
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        # Acquire lock for thread-safe operation | 获取锁以进行线程安全操作
        with self.lock:
            # Add message to the history | 将消息添加到历史中
            self.messages.append(message)
            
        logger.debug(f"Message added successfully. Total messages: {len(self.messages)}")
        
    # Real-time incremental updates are currently disabled (kept for future use)
    
    def _trigger_real_time_memory_update(self):
        """
        Trigger real-time memory update based on certain conditions
        根据特定条件触发实时记忆更新
        """
        # Intentionally left blank; incremental memory updates will be re-enabled in the future
        return
    
    def process_memory_incrementally(self):
        """
        Process conversation memory incrementally in real-time
        实时增量处理对话记忆
        """
        logger.info("Starting process_memory_incrementally")
        
        # Check if already processing | 检查是否已在处理
        if self.processing_memory:
            logger.warning("Memory processing already in progress, skipping")
            return
        
        # Set processing flag | 设置处理标志
        self.processing_memory = True
        logger.debug("Set processing_memory flag to True")
        
        try:
            logger.info("Acquiring lock for message copy")
            # Acquire lock for thread-safe operation | 获取锁以进行线程安全操作
            with self.lock:
                # Create a copy of messages to avoid thread issues | 创建消息副本以避免线程问题
                messages_copy = self.messages.copy()
            logger.info(f"Message copy created. Copied {len(messages_copy)} messages")
            
            # Only process user and assistant messages | 只处理用户和助手的消息
            processing_messages = [msg for msg in messages_copy if msg['role'] in ('user', 'assistant')]
            logger.info(f"Filtered processing messages. Found {len(processing_messages)} user/assistant messages")
            
            if processing_messages:
                logger.info("Starting incremental memory update")
                # Incrementally update memory | 增量更新记忆
                self.current_memory = incrementally_update_memory(processing_messages, self.current_memory)
                self.last_memory_update = time.time()
                logger.info("Real-time memory processing completed | 实时记忆处理完成")
            else:
                logger.info("No processing messages found, skipping memory update")
        except Exception as e:
            logger.error(f"Error in real-time memory processing: {e} | 实时记忆处理中出错: {e}", exc_info=True)
        finally:
            # Reset processing flag | 重置处理标志
            self.processing_memory = False
            logger.debug("Reset processing_memory flag to False")
    
    def capture_assistant_delta(self, text_fragment: str) -> None:
        """Capture streaming assistant text before it is committed."""
        if not text_fragment:
            return
        with self.lock:
            self._assistant_buffer.append(text_fragment)

    def clear_assistant_buffer(self) -> None:
        """Clear any cached assistant text without storing it."""
        with self.lock:
            self._assistant_buffer.clear()

    def commit_assistant_message(self) -> str:
        """Commit buffered assistant text as a single message."""
        with self.lock:
            if not self._assistant_buffer:
                return ""
            content = "".join(self._assistant_buffer).strip()
            self._assistant_buffer.clear()

        if content:
            self.add_message("assistant", content)
        return content

    async def process_memory_incrementally_async(self):
        """
        Process conversation memory incrementally in real-time using async/await
        使用异步方式实时增量处理对话记忆
        """
        logger.info("Starting process_memory_incrementally_async")
        
        # Check if already processing | 检查是否已在处理
        if self.processing_memory:
            logger.warning("Memory processing already in progress (async), skipping")
            return
        
        # Set processing flag | 设置处理标志
        self.processing_memory = True
        logger.debug("Set processing_memory flag to True (async)")
        
        try:
            logger.info("Acquiring lock for message copy (async)")
            # Acquire lock for thread-safe operation | 获取锁以进行线程安全操作
            with self.lock:
                # Create a copy of messages to avoid thread issues | 创建消息副本以避免线程问题
                messages_copy = self.messages.copy()
            logger.info(f"Message copy created (async). Copied {len(messages_copy)} messages")
            
            # Only process user and assistant messages | 只处理用户和助手的消息
            processing_messages = [msg for msg in messages_copy if msg['role'] in ('user', 'assistant')]
            logger.info(f"Filtered processing messages (async). Found {len(processing_messages)} user/assistant messages")
            
            if processing_messages:
                logger.info("Starting incremental memory update (async)")
                # Run the CPU-intensive memory processing in a thread pool to avoid blocking the event loop
                # 在线程池中运行CPU密集型的记忆处理，避免阻塞事件循环
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    logger.info("Submitting memory update task to thread pool")
                    self.current_memory = await loop.run_in_executor(
                        executor, 
                        incrementally_update_memory, 
                        processing_messages, 
                        self.current_memory
                    )
                self.last_memory_update = time.time()
                logger.info("Real-time memory processing completed (async) | 实时记忆处理完成（异步）")
            else:
                logger.info("No processing messages found, skipping memory update (async)")
        except Exception as e:
            logger.error(f"Error in real-time memory processing (async): {e} | 实时记忆处理中出错（异步）: {e}", exc_info=True)
        finally:
            # Reset processing flag | 重置处理标志
            self.processing_memory = False
            logger.debug("Reset processing_memory flag to False (async)")
    
    def save_history(self) -> str:
        """
        Save the current conversation history to a file
        保存当前对话历史到文件
        
        Returns:
            str: Path to the saved history file | 保存的历史文件路径
        """
        # Ensure any in-flight assistant message is captured | 确保正在生成的助手消息被保存
        self.commit_assistant_message()

        # Acquire lock for thread-safe operation | 获取锁以进行线程安全操作
        with self.lock:
            data = self.messages.copy()

        try:
            self.history_dir_path.mkdir(parents=True, exist_ok=True)
            with self.history_file_path.open('w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Conversation history saved to {self.history_file_path} | 对话历史已保存到 {self.history_file_path}")
            return str(self.history_file_path)
        except Exception as e:
            print(f"Error saving conversation history: {e} | 保存对话历史时出错: {e}")
            return ""
    
    def process_memory_async(self):
        """
        Process conversation memory asynchronously
        异步处理对话记忆
        
        This method starts a new thread to process the conversation memory
        without blocking the main conversation flow.
        这个方法启动一个新线程来处理对话记忆，而不会阻塞主对话流程。
        """
        # Check if already processing | 检查是否已在处理
        if self.processing_memory:
            print("Memory processing already in progress | 记忆处理已在进行中")
            return
        
        # Start a new thread for memory processing | 启动一个新线程进行记忆处理
        thread = threading.Thread(target=self._process_memory_thread)
        thread.daemon = True  # Die when the main thread dies | 主线程结束时随之结束
        thread.start()
    
    def _process_memory_thread(self):
        """
        Thread function to process conversation memory
        处理对话记忆的线程函数
        """
        # Set processing flag | 设置处理标志
        self.processing_memory = True
        
        try:
            # Save current history | 保存当前历史
            input_file = self.save_history()
            if not input_file:
                print("Failed to save history for memory processing | 无法保存历史以进行记忆处理")
                return
            
            # Generate output filenames | 生成输出文件名
            transcript_file = str(self.transcript_file_path)

            # Process conversation memory | 处理对话记忆
            print("Starting memory processing... | 开始记忆处理...")
            process_conversation_memory(
                input_file=input_file,
                output_transcript_file=transcript_file
            )
            print("Memory processing completed | 记忆处理完成")

            # Refresh in-memory cache with latest content | 使用最新内容刷新内存缓存
            self._load_memory_from_disk()
            
        except Exception as e:
            print(f"Error in memory processing: {e} | 记忆处理中出错: {e}")
        finally:
            # Reset processing flag | 重置处理标志
            self.processing_memory = False


    def _load_memory_from_disk(self):
        """Load persisted memory system prompt into the in-memory cache."""
        try:
            # Small delay to ensure file is fully written
            time.sleep(0.1)
            # Invalidate import caches to force reload of the module
            importlib.invalidate_caches()
            memory = load_memory_from_system_prompt_file()
            if memory:
                with self.lock:
                    self.current_memory = memory
                    self.last_memory_update = time.time()
                    self.pending_prompt_refresh = True
                    logger.info("Loaded memory content from system prompt file")
        except Exception as exc:
            logger.warning("Failed to load memory from system prompt file: %s", exc)
    
    def get_last_user_message(self) -> str:
        """
        Get the last message from the user
        获取用户的最后一条消息
        
        Returns:
            str: Content of the last user message, or empty string if none | 最后一条用户消息的内容，如果没有则返回空字符串
        """
        # Acquire lock for thread-safe operation | 获取锁以进行线程安全操作
        with self.lock:
            # Iterate through messages in reverse order | 反向遍历消息
            for message in reversed(self.messages):
                if message["role"] == "user":
                    return message["content"]
            return ""
    
    def get_last_assistant_message(self) -> str:
        """
        Get the last message from the assistant
        获取助手的最后一条消息
        
        Returns:
            str: Content of the last assistant message, or empty string if none | 最后一条助手消息的内容，如果没有则返回空字符串
        """
        # Acquire lock for thread-safe operation | 获取锁以进行线程安全操作
        with self.lock:
            # Iterate through messages in reverse order | 反向遍历消息
            for message in reversed(self.messages):
                if message["role"] == "assistant":
                    return message["content"]
            return ""
    
    def get_system_prompt_with_memory(self) -> str:
        """
        Get system prompt integrated with current memory content.
        获取整合当前记忆内容的系统提示词
        
        Returns:
            str: System prompt with integrated memory | 整合记忆的系统提示词
        """
        return update_system_prompt_with_current_memory(self)
    
    def get_current_memory(self) -> dict:
        """
        Get current memory content
        获取当前记忆内容
        
        Returns:
            dict: Current memory content | 当前记忆内容
        """
        with self.lock:
            return deepcopy(self.current_memory)
    
    def clear_history(self):
        """
        Clear the conversation history
        清除对话历史
        """
        # Acquire lock for thread-safe operation | 获取锁以进行线程安全操作
        with self.lock:
            # Clear messages | 清除消息
            self.messages.clear()
            self._assistant_buffer.clear()
            self._running_summary = None
            print("Conversation history cleared | 对话历史已清除")

    def summarize_recent_conversation(
        self,
        *,
        max_tokens_before_summary: int | None = None,
        max_summary_tokens: int = 128,
    ) -> str:
        """Summarize the latest conversation and compress history to the summary."""
        self.commit_assistant_message()

        with self.lock:
            messages_copy = [dict(message) for message in self.messages]

        if len(messages_copy) <= 1:
            logger.info("Skipping summary; no conversation content beyond system message")
            return ""

        summary_text, updated_running_summary = generate_summary(
            messages_copy,
            max_tokens_before_summary=max_tokens_before_summary,
            max_summary_tokens=max_summary_tokens,
            running_summary=self._running_summary,
        )
        if not summary_text:
            logger.warning("Summarizer returned empty content")
            return ""

        self._replace_history_with_summary(summary_text)
        self._running_summary = updated_running_summary
        return summary_text

    def _replace_history_with_summary(self, summary_text: str) -> None:
        """Replace stored messages with system prompt and summary output."""
        now_iso = datetime.now().isoformat()

        with self.lock:
            system_content = ""
            for message in self.messages:
                if message.get("role") == "system":
                    system_content = message.get("content", "")
                    break

        if not system_content:
            memory_snapshot = self.get_current_memory()
            system_content = generate_system_prompt_with_memory(memory_snapshot)

        new_messages = [
            {
                "role": "system",
                "content": system_content,
                "timestamp": now_iso,
            },
            {
                "role": "summariser output",
                "content": summary_text,
                "timestamp": now_iso,
            },
        ]

        with self.lock:
            self.messages = new_messages
            self._assistant_buffer.clear()

        logger.info("Conversation history compressed into summary (%d chars)", len(summary_text))

    def refresh_system_message(self, prompt_text: str) -> None:
        """Update or insert the system prompt message with the latest memory."""
        with self.lock:
            for message in self.messages:
                if message.get("role") == "system":
                    message["content"] = prompt_text
                    message["timestamp"] = datetime.now().isoformat()
                    break
            else:
                self.messages.insert(0, {
                    "role": "system",
                    "content": prompt_text,
                    "timestamp": datetime.now().isoformat()
                })


# Global instance for the application
# 应用程序的全局实例
conversation_history = ConversationHistoryManager()


def initialize_history_with_memory(memory_file: Optional[str] = None):
    """
    Initialize conversation history with existing memory
    使用现有记忆初始化对话历史
    
    Args:
        memory_file: Path to existing memory file | 现有记忆文件的路径
    """
    global conversation_history
    
    # If a memory file is provided, load it | 如果提供了记忆文件，则加载它
    if memory_file and os.path.exists(memory_file):
        try:
            with open(memory_file, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
            
            # Check if it's in the messages format | 检查是否为消息格式
            if isinstance(memory_data, list):
                # Directly load messages | 直接加载消息
                conversation_history.messages = memory_data
                conversation_history._running_summary = None
                print("Initialized conversation with existing messages | 使用现有消息初始化对话")
            elif isinstance(memory_data, dict) and "messages" in memory_data:
                # Load messages from dict | 从字典中加载消息
                conversation_history.messages = memory_data["messages"]
                conversation_history._running_summary = None
                print("Initialized conversation with existing messages | 使用现有消息初始化对话")
            else:
                # Add default system message | 添加默认系统消息
                conversation_history.add_message(
                    "system", 
                    str(memory_data) if memory_data else "You are a personal assistant Xiao Yun. Please answer user questions accurately and kindly."
                )
                conversation_history._running_summary = None
                print("Initialized conversation with existing memory content | 使用现有记忆内容初始化对话")
        except Exception as e:
            print(f"Error loading memory file: {e} | 加载记忆文件时出错: {e}")
    else:
        # Add default or memory-backed system prompt | 添加默认或基于记忆的系统提示词
        memory_content = conversation_history.get_current_memory()
        system_prompt_text = generate_system_prompt_with_memory(memory_content)
        with conversation_history.lock:
            conversation_history.messages.clear()
            conversation_history._running_summary = None
        conversation_history.add_message("system", system_prompt_text)


def save_and_process_conversation():
    """
    Save the current conversation and process it for memory extraction
    保存当前对话并处理以提取记忆
    """
    global conversation_history
    
    # Process memory asynchronously | 异步处理记忆
    conversation_history.process_memory_async()


