# dashscope SDK version must be no less than 1.23.9 | dashscope SDK 版本需不低于 1.23.9
import os
import base64
import signal
import sys
import time
import pyaudio
import contextlib
import threading
import queue
from dashscope.audio.qwen_omni import *
import dashscope

# API Key for Singapore and Beijing regions are different. Get API Key: https://www.alibabacloud.com/help/zh/model-studio/get-api-key
# 新加坡和北京地域的API Key不同。获取API Key：https://www.alibabacloud.com/help/zh/model-studio/get-api-key
# If environment variable is not set, replace the following line with dashscope.api_key = "sk-xxx"
# 如果没有设置环境变量，请用您的 API Key 将下行替换为dashscope.api_key = "sk-xxx"
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

# Default voice setting | 默认语音设置
voice = 'Jennifer'
conversation = None

# Wake words and exit words settings | 唤醒词和退出词设置
WAKE_WORDS = [word.strip() for word in os.getenv('WAKE_WORDS', '你好灵灵,灵灵你好').split(',')]
EXIT_WORDS = [word.strip() for word in os.getenv('EXIT_WORDS', '再见灵灵,灵灵再见').split(',')]

# Flag to track if system is activated | 跟踪系统是否被激活的标志
is_activated = False

# Import conversation history management
# 导入对话历史管理
from conversation_history import conversation_history, initialize_history_with_memory, save_and_process_conversation
# Import system prompt management
# 导入系统提示词管理
from system_prompt import update_system_prompt_with_current_memory


class B64PCMPlayer:
    def __init__(self, pya: pyaudio.PyAudio, sample_rate=24000, chunk_size_ms=100):
        self.pya = pya
        self.sample_rate = sample_rate
        self.chunk_size_bytes = chunk_size_ms * sample_rate * 2 // 1000
        self.player_stream = pya.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=sample_rate,
                                      output=True)

        self.raw_audio_buffer: queue.Queue = queue.Queue()
        self.b64_audio_buffer: queue.Queue = queue.Queue()
        self.status_lock = threading.Lock()
        self.status = 'playing'
        self.decoder_thread = threading.Thread(target=self.decoder_loop)
        self.player_thread = threading.Thread(target=self.player_loop)
        self.decoder_thread.start()
        self.player_thread.start()
        self.complete_event: threading.Event = None

    def decoder_loop(self):
        # Decode loop: Decode Base64 audio data into raw PCM data | 解码循环：将Base64音频数据解码为原始PCM数据
        while self.status != 'stop':
            recv_audio_b64 = None
            with contextlib.suppress(queue.Empty):
                recv_audio_b64 = self.b64_audio_buffer.get(timeout=0.1)
            if recv_audio_b64 is None:
                continue
            recv_audio_raw = base64.b64decode(recv_audio_b64)
            # Push raw audio data into queue, process in chunks | 将原始音频数据推入队列，按块处理
            for i in range(0, len(recv_audio_raw), self.chunk_size_bytes):
                chunk = recv_audio_raw[i:i + self.chunk_size_bytes]
                self.raw_audio_buffer.put(chunk)

    def player_loop(self):
        # Playback loop: Read PCM data from buffer and play through audio device | 播放循环：从缓冲区读取PCM数据并通过音频设备播放
        while self.status != 'stop':
            recv_audio_raw = None
            with contextlib.suppress(queue.Empty):
                recv_audio_raw = self.raw_audio_buffer.get(timeout=0.1)
            if recv_audio_raw is None:
                if self.complete_event:
                    self.complete_event.set()
                continue
                # Write chunk to pyaudio player, wait for chunk to finish playing | 将块写入pyaudio音频播放器，等待播放完这个块
            self.player_stream.write(recv_audio_raw)

    def cancel_playing(self):
        # Cancel current playback | 取消当前播放
        self.b64_audio_buffer.queue.clear()
        self.raw_audio_buffer.queue.clear()

    def add_data(self, data):
        # Add Base64 encoded audio data to playback queue | 添加Base64编码的音频数据到播放队列
        self.b64_audio_buffer.put(data)

    def wait_for_complete(self):
        # Wait for playback to complete | 等待播放完成
        self.complete_event = threading.Event()
        self.complete_event.wait()
        self.complete_event = None

    def shutdown(self):
        # Shut down the player | 关闭播放器
        self.status = 'stop'
        self.decoder_thread.join()
        self.player_thread.join()
        self.player_stream.close()


def check_wake_words(transcript, wake_words):
    """
    Check if any wake word is present in the transcript
    检查转录文本中是否包含任何唤醒词
    
    Args:
        transcript: The user's speech transcript | 用户的语音转录
        wake_words: List of wake words | 唤醒词列表
        
    Returns:
        bool: True if any wake word is detected, False otherwise | 如果检测到任何唤醒词则返回True，否则返回False
    """
    for word in wake_words:
        if word in transcript:
            return True
    return False


def check_exit_words(transcript, exit_words):
    """
    Check if any exit word is present in the transcript
    检查转录文本中是否包含任何退出词
    
    Args:
        transcript: The user's speech transcript | 用户的语音转录
        exit_words: List of exit words | 退出词列表
        
    Returns:
        bool: True if any exit word is detected, False otherwise | 如果检测到任何退出词则返回True，否则返回False
    """
    for word in exit_words:
        if word in transcript:
            return True
    return False


class MyCallback(OmniRealtimeCallback):
    def on_open(self) -> None:
        global pya
        global mic_stream
        global b64_player
        print('connection opened, init microphone | 连接已建立，初始化麦克风')
        pya = pyaudio.PyAudio()
        mic_stream = pya.open(format=pyaudio.paInt16,
                              channels=1,
                              rate=16000,
                              input=True)
        b64_player = B64PCMPlayer(pya)

    def on_close(self, close_status_code, close_msg) -> None:
        print('connection closed with code: {}, msg: {}, destroy microphone | 连接已关闭，状态码: {}, 消息: {}，销毁麦克风'.format(close_status_code, close_msg, close_status_code, close_msg))
        sys.exit(0)

    def on_event(self, response: str) -> None:
        # 声明全局变量
        global conversation
        global b64_player
        global is_activated
        global WAKE_WORDS
        global EXIT_WORDS
        
        try:
            type = response['type']
            if 'session.created' == type:
                print('start session: {} | 开始会话: {}'.format(response['session']['id'], response['session']['id']))
            if 'conversation.item.input_audio_transcription.completed' == type:
                transcript = response['transcript']
                print('question: {} | 问题: {}'.format(transcript, transcript))
                # Check for wake words | 检查唤醒词
                if check_wake_words(transcript, WAKE_WORDS) and not is_activated:
                    print('Wake word detected! Activating system... | 检测到唤醒词！激活系统...')
                    is_activated = True
                    # Add user message to conversation history
                    # 将用户消息添加到对话历史
                    conversation_history.add_message("user", transcript)
                # Check for exit words | 检查退出词
                elif check_exit_words(transcript, EXIT_WORDS) and is_activated:
                    print('Exit word detected! Deactivating system... | 检测到退出词！停用系统...')
                    is_activated = False
                    print('System deactivated. Say any wake word to reactivate. | 系统已停用。说出任意唤醒词重新激活。')
                # Process regular conversation when activated | 激活时处理正常对话
                elif is_activated:
                    # Add user message to conversation history
                    # 将用户消息添加到对话历史
                    conversation_history.add_message("user", transcript)
            if 'response.audio_transcript.delta' == type:
                text = response['delta']
                print("got llm response delta: {} | 收到LLM响应文本: {}".format(text, text))
            if 'response.audio.delta' == type:
                recv_audio_b64 = response['delta']
                # Only play audio when system is activated | 仅在系统激活时播放音频
                if is_activated:
                    b64_player.add_data(recv_audio_b64)
            if 'input_audio_buffer.speech_started' == type:
                print('======VAD Speech Start====== | ======语音开始======')
                # Only cancel playing when system is activated | 仅在系统激活时取消播放
                if is_activated:
                    b64_player.cancel_playing()
            if 'response.done' == type:
                print('======RESPONSE DONE====== | ======响应完成======')
                print('[Metric] response: {}, first text delay: {}ms, first audio delay: {}ms | [指标] 响应ID: {}, 首文本延迟: {}ms, 首音频延迟: {}ms'.format(
                                conversation.get_last_response_id(), 
                                conversation.get_last_first_text_delay(), 
                                conversation.get_last_first_audio_delay(),
                                conversation.get_last_response_id(), 
                                conversation.get_last_first_text_delay(), 
                                conversation.get_last_first_audio_delay()))
                # Only process memory and update conversation when activated | 仅在激活时处理记忆和更新对话
                if is_activated:
                    # Add assistant message to conversation history
                    # 将助手消息添加到对话历史
                    # Note: In this version, we don't have the full assistant message, but in a more complete implementation
                    # we would collect and add it here.
                    # 注意：在这个版本中，我们没有完整的助手消息，但在更完整的实现中，我们会在这里收集并添加它。
                    
                    # Trigger real-time memory processing
                    # 触发实时记忆处理
                    conversation_history.process_memory_incrementally()
                    
                    # Update session with new system prompt if memory was updated
                    # 如果记忆已更新，则使用新的系统提示词更新会话
                    import time
                    if conversation and hasattr(conversation_history, 'last_memory_update') and \
                       (time.time() - conversation_history.last_memory_update) < 5:  # Within 5 seconds
                        # 获取包含更新记忆的系统提示词
                        system_prompt = update_system_prompt_with_current_memory(conversation_history)
                        # 打印系统提示内容
                        print("===== SYSTEM PROMPT UPDATE START =====")
                        print("Current system prompt:")
                        print(system_prompt)
                        print("===== SYSTEM PROMPT UPDATE END =====")
                        # 更新会话配置
                        conversation.update_session(instructions=system_prompt)
                        print("Session updated with new memory content | 会话已使用新记忆内容更新")
        except Exception as e:
            print('[Error] {} | [错误] {}'.format(e, e))
            return

