# Integration Guide for save_memory_langmem Module
# save_memory_langmem模块集成指南

This guide explains how to integrate the save_memory_langmem module with the Qwen-Omni real-time conversation system.
本指南说明如何将save_memory_langmem模块与Qwen-Omni实时对话系统集成。

## Overview | 概述

The integration adds conversation history tracking and memory processing capabilities to the real-time conversation system. 
集成增加了对话历史跟踪和记忆处理功能到实时对话系统中。

Key features | 主要功能:
- Automatic conversation history collection | 自动收集对话历史
- Asynchronous memory processing | 异步记忆处理
- Integration with langmem system for user profile and segment extraction | 与langmem系统集成以提取用户档案和片段
- Persistent storage of conversation history | 对话历史的持久化存储

## Architecture | 架构

The integration is implemented through the following components:
集成通过以下组件实现：

1. **conversation_history.py**: New module for conversation history management
   **conversation_history.py**: 用于对话历史管理的新模块
2. **ConversationHistoryManager**: Class that manages conversation messages and memory processing
   **ConversationHistoryManager**: 管理对话消息和记忆处理的类
3. **Integration with app.py**: Modifications to the main application to use conversation history
   **与app.py集成**: 修改主应用程序以使用对话历史

## Integration Details | 集成详情

### 1. Conversation History Module | 对话历史模块

The `conversation_history.py` module provides:
`conversation_history.py`模块提供：

- **ConversationHistoryManager class**: Manages conversation messages and memory processing
  **ConversationHistoryManager类**: 管理对话消息和记忆处理
- **Global instance**: A global instance `conversation_history` for application-wide use
  **全局实例**: 用于应用程序范围使用的全局实例`conversation_history`
- **Memory processing functions**: Functions to initialize with existing memory and process conversations
  **记忆处理函数**: 使用现有记忆初始化和处理对话的函数

### 2. Modifications to app.py | 对app.py的修改

Key modifications in `app.py`:
`app.py`中的主要修改：

- **Import statement**: Import conversation history management
  **导入语句**: 导入对话历史管理
- **Message tracking**: Track user questions and assistant responses
  **消息跟踪**: 跟踪用户问题和助手响应
- **Signal handling**: Process memory on Ctrl+C or connection close
  **信号处理**: 在Ctrl+C或连接关闭时处理记忆
- **Initialization**: Initialize with existing memory
  **初始化**: 使用现有记忆初始化

### 3. Memory Processing Workflow | 记忆处理工作流程

1. **Conversation Collection**: Messages are collected during the conversation
   **对话收集**: 在对话期间收集消息
2. **History Saving**: Conversation history is saved to a JSON file when processing is triggered
   **历史保存**: 触发处理时将对话历史保存到JSON文件
3. **Asynchronous Processing**: Memory processing runs in a separate thread to avoid blocking the conversation
   **异步处理**: 记忆处理在单独的线程中运行，以避免阻塞对话
4. **Memory Extraction**: The langmem system extracts user profiles and conversation segments
   **记忆提取**: langmem系统提取用户档案和对话片段
5. **Persistent Storage**: Processed memory is saved for future use
   **持久化存储**: 处理后的记忆被保存以供将来使用

## Usage | 使用方法

### Running the Integrated System | 运行集成系统

1. Ensure all dependencies are installed
   确保所有依赖项已安装:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables
   设置环境变量:
   ```bash
   export DASHSCOPE_API_KEY=your_api_key_here
   export QWEN_API_KEY=your_qwen_api_key_here  # For langmem processing
   ```

3. Run the application
   运行应用程序:
   ```bash
   python app.py
   ```

### Memory Processing | 记忆处理

Memory processing happens automatically in the following situations:
在以下情况下自动进行记忆处理：

- When the conversation is stopped with Ctrl+C
  当使用Ctrl+C停止对话时
- When the WebSocket connection is closed
  当WebSocket连接关闭时

The processing is asynchronous and does not block the conversation flow.
处理是异步的，不会阻塞对话流程。

### Output Files | 输出文件

Processed memory files are saved in the `history` directory:
处理后的记忆文件保存在`history`目录中：

- `conversation_history_*.json`: Raw conversation history
  `conversation_history_*.json`: 原始对话历史
- `conversation_history_*_transcript.json`: Conversation transcript
  `conversation_history_*_transcript.json`: 对话记录
- `conversation_history_*_memory.json`: Processed memory with user profile and segments
  `conversation_history_*_memory.json`: 处理后的记忆，包含用户档案和片段

## Customization | 自定义

### Changing History File Location | 更改历史文件位置

Modify the `history_dir` parameter in `ConversationHistoryManager`:
修改`ConversationHistoryManager`中的`history_dir`参数:

```python
conversation_history = ConversationHistoryManager(history_file_prefix="my_prefix")
```

### Adjusting Processing Behavior | 调整处理行为

The memory processing can be customized by modifying the `process_memory_async` method in `conversation_history.py`.
可以通过修改`conversation_history.py`中的`process_memory_async`方法来自定义记忆处理。

## Troubleshooting | 故障排除

### Memory Processing Failures | 记忆处理失败

If memory processing fails, check:
如果记忆处理失败，请检查：

1. Environment variables are set correctly
   环境变量设置正确
2. The langmem system is properly installed and configured
   langmem系统已正确安装和配置
3. Input files are in the correct format
   输入文件格式正确

### Performance Issues | 性能问题

If there are performance issues:
如果存在性能问题：

1. Check that memory processing is running asynchronously
   检查记忆处理是否异步运行
2. Ensure the conversation history is not excessively large
   确保对话历史不会过大

## Future Improvements | 未来改进

Potential improvements to the integration:
集成的潜在改进：

1. **Incremental Memory Processing**: Process conversation segments incrementally rather than the entire conversation
   **增量记忆处理**: 增量处理对话片段而不是整个对话
2. **Memory-based Personalization**: Use extracted user profiles to personalize future conversations
   **基于记忆的个性化**: 使用提取的用户档案个性化未来的对话
3. **Enhanced Error Handling**: Add more robust error handling for memory processing failures
   **增强的错误处理**: 为记忆处理失败添加更强大的错误处理
4. **Performance Optimization**: Optimize memory processing for large conversations
   **性能优化**: 优化大对话的记忆处理