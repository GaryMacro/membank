import os
import base64
import signal
import sys
import time
import pyaudio
import contextlib
import threading
import queue
import asyncio
import logging
import wave
import glob
from typing import Any, Dict
from dashscope.audio.qwen_omni import *
import dashscope

# Configure logging
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app_system.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

from config import (
    DASHSCOPE_API_KEY, 
    REGION, 
    DEFAULT_VOICE, 
    WEBSOCKET_URLS,
    INPUT_AUDIO_FORMAT,
    OUTPUT_AUDIO_FORMAT,
    WAKE_WORDS,
    EXIT_WORDS
)

# Import conversation history management
# 导入对话历史管理
from conversation_history import conversation_history, initialize_history_with_memory, save_and_process_conversation
# Import system prompt management
# 导入系统提示词管理
from system_prompt import update_system_prompt_with_current_memory
from session_state import SessionState

# Set API key | 设置API密钥
# 从配置文件中获取并设置DashScope API密钥
dashscope.api_key = DASHSCOPE_API_KEY

# Voice settings | 语音设置
# 设置默认语音，用于语音合成
voice = DEFAULT_VOICE
# 初始化对话对象，将在后续初始化
conversation = None

# Wake words and exit words settings | 唤醒词和退出词设置
wake_words = WAKE_WORDS
exit_words = EXIT_WORDS

# Flag to track if system is activated | 跟踪系统是否被激活的标志
is_activated = False

# Session state manager for prompt refresh | 会话状态管理器，用于刷新提示词
session_state = SessionState(conversation_history, update_system_prompt_with_current_memory)

# Textual assistant streaming events | 助手文本流事件类型
TEXT_STREAM_EVENTS = {
    'response.output_text.delta',
    'response.audio_transcript.delta',
    'response.text.delta',
}

def handle_exit_sequence(transcript: str) -> None:
    """Handle exit word detection by saving history and triggering memory processing."""
    global is_activated

    if not is_activated:
        return

    print(f'\n[EXIT WORD DETECTED] System deactivated! | [检测到退出词] 系统已停用！')
    print('System is now sleeping. Say any wake word to reactivate. | 系统正在休眠。说出任意唤醒词重新激活。')

    # Record the final user message before processing memory
    conversation_history.add_message("user", transcript)

    # Start asynchronous memory processing so profile/segments update immediately after exit
    save_and_process_conversation()

    summary = conversation_history.summarize_recent_conversation()
    if summary:
        print("[SUMMARY] Previous conversation compressed:")
        print(summary)

    # Reset any buffered assistant response snippets
    conversation_history.clear_assistant_buffer()

    is_activated = False


# 添加录音函数
def record_audio(duration=20, sample_rate=16000, chunk_size=1024):
    """
    录制指定时长的音频
    
    Args:
        duration: 录音时长（秒），默认20秒
        sample_rate: 采样率，默认16000Hz
        chunk_size: 音频块大小，默认1024
    
    Returns:
        None
    """
    # 确保 recordings 目录存在
    if not os.path.exists("recordings"):
        os.makedirs("recordings")
    
    # 删除已有的录音文件，确保只保留一个录音文件
    existing_files = glob.glob("recordings/*.wav")
    for file in existing_files:
        try:
            os.remove(file)
            print(f"[RECORDING] 已删除旧录音文件: {file}")
        except Exception as e:
            print(f"[RECORDING] 删除旧录音文件失败 {file}: {e}")
    
    # 获取当前时间戳作为文件名
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"recordings/recording_{timestamp}.wav"
    
    print(f"[RECORDING] 开始录音，时长: {duration}秒")
    
    # 初始化PyAudio
    p = pyaudio.PyAudio()
    
    # 打开音频流
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)
    
    # 存储录制的音频帧
    frames = []
    
    # 计算需要录制的帧数
    total_frames = int(sample_rate / chunk_size * duration)
    
    # 录制音频
    for i in range(total_frames):
        data = stream.read(chunk_size)
        frames.append(data)
    
    # 停止并关闭音频流
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # 保存为WAV文件
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    print(f"[RECORDING] 录音已保存至: {filename}")


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


class B64PCMPlayer:
    """
    Base64 Encoded PCM Audio Player Class
    Base64编码的PCM音频播放器类
    
    这个类实现了Base64编码的PCM音频数据的解码和播放功能。
    它使用两个线程分别处理音频解码和播放，以确保流畅的音频输出。
    """
    def __init__(self, pya: pyaudio.PyAudio, sample_rate=24000, chunk_size_ms=100):
        """
        初始化音频播放器
        
        Args:
            pya: PyAudio实例，用于音频播放
            sample_rate: 采样率，默认24000Hz
            chunk_size_ms: 音频块大小（毫秒），默认100ms
        """
        # 保存PyAudio实例
        self.pya = pya
        # 设置音频采样率
        self.sample_rate = sample_rate
        # 计算音频块大小（字节）
        self.chunk_size_bytes = chunk_size_ms * sample_rate * 2 // 1000
        # 创建PyAudio音频流用于播放
        self.player_stream = pya.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=sample_rate,
                                      output=True)

        # 创建原始音频数据缓冲区队列
        self.raw_audio_buffer: queue.Queue = queue.Queue()
        # 创建Base64编码音频数据缓冲区队列
        self.b64_audio_buffer: queue.Queue = queue.Queue()
        # 创建状态锁，用于线程安全的状态访问
        self.status_lock = threading.Lock()
        # 设置播放器状态为播放中
        self.status = 'playing'
        # 创建解码线程
        self.decoder_thread = threading.Thread(target=self.decoder_loop)
        # 创建播放线程
        self.player_thread = threading.Thread(target=self.player_loop)
        # 启动解码线程
        self.decoder_thread.start()
        # 启动播放线程
        self.player_thread.start()
        # 初始化播放完成事件对象
        self.complete_event: threading.Event | None = None

        # Add playback status flag | 添加播放状态标志
        # 添加播放状态标志，用于跟踪当前是否正在播放音频
        self.is_playing = False
        # 创建播放锁，用于线程安全的播放状态访问
        self.playback_lock = threading.Lock()

    def decoder_loop(self):
        """Decoder loop: Decode Base64 audio data into raw PCM data
        解码循环：将Base64音频数据解码为原始PCM数据
        
        这个方法在独立线程中运行，持续从Base64音频数据缓冲区
        读取数据并解码为原始PCM数据，然后放入原始音频数据缓冲区。
        """
        # 当播放器状态不为停止时持续运行
        while self.status != 'stop':
            # 尝试从Base64音频数据缓冲区获取数据
            recv_audio_b64 = None
            with contextlib.suppress(queue.Empty):
                # 设置超时时间为0.1秒，避免线程阻塞
                recv_audio_b64 = self.b64_audio_buffer.get(timeout=0.1)
            # 如果没有获取到数据，则继续下一次循环
            if recv_audio_b64 is None:
                continue
            # 将Base64编码的音频数据解码为原始PCM数据
            recv_audio_raw = base64.b64decode(recv_audio_b64)
            # Push raw audio data into queue, process in chunks | 将原始音频数据推入队列，按块处理
            # 将原始PCM数据按块分割并放入原始音频数据缓冲区
            for i in range(0, len(recv_audio_raw), self.chunk_size_bytes):
                # 分割音频数据为块
                chunk = recv_audio_raw[i:i + self.chunk_size_bytes]
                # 将音频块放入原始音频数据缓冲区
                self.raw_audio_buffer.put(chunk)

    def player_loop(self):
        """Playback loop: Read PCM data from buffer and play through audio device
        播放循环：从缓冲区读取PCM数据并通过音频设备播放
        
        这个方法在独立线程中运行，持续从原始音频数据缓冲区
        读取数据并通过音频设备播放。
        """
        # 当播放器状态不为停止时持续运行
        while self.status != 'stop':
            # 尝试从原始音频数据缓冲区获取数据
            recv_audio_raw = None
            with contextlib.suppress(queue.Empty):
                # 设置超时时间为0.1秒，避免线程阻塞
                recv_audio_raw = self.raw_audio_buffer.get(timeout=0.1)
            # 如果没有获取到数据
            if recv_audio_raw is None:
                # Set status after playback completes | 播放完成后设置状态
                # 获取播放锁并设置播放状态为False
                with self.playback_lock:
                    self.is_playing = False
                # 如果存在播放完成事件，则设置该事件
                if self.complete_event:
                    self.complete_event.set()
                # 继续下一次循环
                continue
            
            # Set status when starting playback | 开始播放时设置状态
            # 获取播放锁并设置播放状态为True
            with self.playback_lock:
                self.is_playing = True
                
            # Write chunk to pyaudio player, wait for chunk to finish playing | 将块写入pyaudio音频播放器，等待播放完这个块
            # 将音频块写入PyAudio播放流进行播放
            self.player_stream.write(recv_audio_raw)

    def cancel_playing(self):
        """Cancel current playback | 取消当前播放
        
        清空所有音频数据缓冲区，立即停止当前播放。
        """
        # 清空Base64音频数据缓冲区
        self.b64_audio_buffer.queue.clear()
        # 清空原始音频数据缓冲区
        self.raw_audio_buffer.queue.clear()
        
        # 获取播放锁并设置播放状态为False
        with self.playback_lock:
            self.is_playing = False

    def add_data(self, data):
        """Add Base64 encoded audio data to playback queue
        添加Base64编码的音频数据到播放队列
        
        Args:
            data: Base64编码的音频数据
        """
        # 将Base64编码的音频数据放入缓冲区队列
        self.b64_audio_buffer.put(data)

    def wait_for_complete(self):
        """Wait for playback to complete | 等待播放完成
        
        等待当前音频播放完成。
        """
        # 创建并设置播放完成事件
        self.complete_event = threading.Event()
        # 等待事件被触发（播放完成）
        self.complete_event.wait()
        # 重置播放完成事件
        self.complete_event = None

    def shutdown(self):
        """Shut down the player | 关闭播放器
        
        安全地关闭播放器，停止所有线程并释放资源。
        """
        # 设置播放器状态为停止
        self.status = 'stop'
        # 等待解码线程结束
        self.decoder_thread.join()
        # 等待播放线程结束
        self.player_thread.join()
        # 关闭PyAudio播放流
        self.player_stream.close()
        
    def is_currently_playing(self):
        """Check if currently playing audio | 检查当前是否正在播放音频
        
        Returns:
            bool: 如果正在播放音频则返回True，否则返回False
        """
        # 获取播放锁并返回播放状态
        with self.playback_lock:
            return self.is_playing


class MyCallback(OmniRealtimeCallback):
    """
    Callback class for handling real-time events from Qwen-Omni
    Qwen-Omni实时事件回调处理类
    
    This class handles various events during the real-time conversation,
    including connection status, speech recognition results, audio playback,
    and VAD (Voice Activity Detection) events.
    这个类处理实时对话过程中的各种事件，
    包括连接状态、语音识别结果、音频播放和VAD（语音活动检测）事件。
    """
    
    def on_open(self) -> None:
        """
        Handle connection open event
        处理连接打开事件
        
        This method is called when the WebSocket connection is successfully established.
        It initializes the microphone and audio player.
        当WebSocket连接成功建立时调用此方法。
        它初始化麦克风和音频播放器。
        """
        global pya
        global mic_stream
        global b64_player
        logger.info('Connection opened, initializing microphone | 连接已建立，初始化麦克风')
        pya = pyaudio.PyAudio()
        mic_stream = pya.open(format=pyaudio.paInt16,
                              channels=1,
                              rate=16000,
                              input=True)
        b64_player = B64PCMPlayer(pya)
        
        # Print welcome message with wake word information
        # 打印包含唤醒词信息的欢迎消息
        print("=" * 50)
        print("Qwen-Omni Real-time Conversation System")
        print("Qwen-Omni实时对话系统")
        print("-" * 50)
        print(f"Wake words: {', '.join(wake_words)} | 唤醒词: {', '.join(wake_words)}")
        print(f"Exit words: {', '.join(exit_words)} | 退出词: {', '.join(exit_words)}")
        print("System is waiting for wake word... | 系统正在等待唤醒词...")
        print("=" * 50)

    def on_close(self, close_status_code, close_msg) -> None:
        """
        Handle connection close event
        处理连接关闭事件
        
        This method is called when the WebSocket connection is closed.
        It cleans up resources and exits the program.
        当WebSocket连接关闭时调用此方法。
        它清理资源并退出程序。
        
        Args:
            close_status_code: Close status code | 关闭状态码
            close_msg: Close message | 关闭消息
        """
        logger.info('Connection closed with code: {}, msg: {}, destroying microphone | 连接已关闭，状态码: {}, 消息: {}，销毁麦克风'.format(close_status_code, close_msg, close_status_code, close_msg))
        sys.exit(0)

    def on_event(self, message: Any) -> None:
        """
        Handle real-time events from Qwen-Omni
        处理来自Qwen-Omni的实时事件
        
        This method handles various real-time events during the conversation,
        including session creation, transcription results, audio playback,
        and VAD events.
        此方法处理对话过程中的各种实时事件，
        包括会话创建、转录结果、音频播放和VAD事件。
        
        Args:
            response: Event response data | 事件响应数据
        """
        global conversation
        global b64_player
        global is_activated
        global wake_words
        global exit_words
        global session_state

        try:
            event_type = message.get('type') if isinstance(message, dict) else None
            if not event_type:
                return
            
            # Handle session creation event
            # 处理会话创建事件
            if 'session.created' == event_type:
                session_id = message.get('session', {}).get('id')
                logger.info('Start session: {} | 开始会话: {}'.format(session_id, session_id))
            
            # Handle audio transcription completion event
            # 处理音频转录完成事件
            if 'conversation.item.input_audio_transcription.completed' == event_type:
                transcript = message.get('transcript', '')
                logger.info('Question: {} | 问题: {}'.format(transcript, transcript))
                
                # Check for wake words | 检查唤醒词
                if check_wake_words(transcript, wake_words) and not is_activated:
                    print(f'\n[WAKE WORD DETECTED] System activated! | [检测到唤醒词] 系统已激活！')
                    print(f'Welcome! I am Xiao Yun, your personal assistant. How can I help you? | 欢迎！我是灵灵，您的个人助手。我怎样可以帮助您？')
                    is_activated = True
                    
                    # 在系统激活后开始录音
                    # 在新线程中执行录音，避免阻塞主线程
                    recording_thread = threading.Thread(target=record_audio, args=(20,))
                    recording_thread.daemon = True
                    recording_thread.start()
                    
                    # Add user message to conversation history
                    # 将用户消息添加到对话历史
                    conversation_history.add_message("user", transcript)
                    session_state.refresh_if_needed("wake-word")
                
                # Check for exit words | 检查退出词
                elif check_exit_words(transcript, exit_words) and is_activated:
                    handle_exit_sequence(transcript)
                
                # Process regular conversation when activated | 激活时处理正常对话
                elif is_activated:
                    # Add user message to conversation history
                    # 将用户消息添加到对话历史
                    conversation_history.add_message("user", transcript)
            
            # Handle incremental audio transcription event
            # 处理增量音频转录事件
            if event_type in TEXT_STREAM_EVENTS and is_activated:
                delta_text = message.get('delta')
                if isinstance(delta_text, str) and delta_text.strip():
                    conversation_history.capture_assistant_delta(delta_text)
                if delta_text:
                    logger.debug("Got LLM response delta: {} | 收到LLM响应文本: {}".format(delta_text, delta_text))
            
            # Handle audio data event
            # 处理音频数据事件
            if 'response.audio.delta' == event_type:
                recv_audio_b64 = message.get('delta')
                # Only play audio when system is activated | 仅在系统激活时播放音频
                if is_activated:
                    b64_player.add_data(recv_audio_b64)
            
            # Handle speech start event (VAD)
            # 处理语音开始事件（VAD）
            if 'input_audio_buffer.speech_started' == event_type:
                logger.info('======VAD Speech Start====== | ======语音开始======')
                # Only cancel playing when system is activated | 仅在系统激活时取消播放
                if is_activated:
                    b64_player.cancel_playing()
            
            # Handle response completion event
            # 处理响应完成事件
            if 'response.done' == event_type:
                logger.info('======RESPONSE DONE====== | ======响应完成======')
                if conversation is not None:
                    logger.info('[Metric] response: %s | first text delay: %sms | first audio delay: %sms',
                                getattr(conversation, 'get_last_response_id', lambda: 'unknown')(),
                                getattr(conversation, 'get_last_first_text_delay', lambda: 'unknown')(),
                                getattr(conversation, 'get_last_first_audio_delay', lambda: 'unknown')())
                
                # Only process memory and update conversation when activated | 仅在激活时处理记忆和更新对话
                if is_activated:
                    committed_text = conversation_history.commit_assistant_message()
                    if committed_text:
                        logger.info("Assistant response captured (chars=%d) | 助手响应已记录(字符数=%d)", len(committed_text), len(committed_text))

                session_state.refresh_if_needed("response-done")
                        
        except Exception as e:
            logger.error('[Error] {} | [错误] {}'.format(e, e))
            return


def signal_handler(sig, frame):
    """Signal handler for graceful program shutdown | 信号处理器，用于优雅地关闭程序
    
    处理Ctrl+C信号，安全地关闭程序。
    
    Args:
        sig: 信号编号
        frame: 当前堆栈帧
    """
    # 打印程序停止信息
    print('\nCtrl+C detected, stopping conversation... | 检测到 Ctrl+C，正在停止对话 ...')
    try:
        # 保存并处理对话历史
        print('Saving conversation history and processing memory... | 保存对话历史并处理记忆...')
        save_and_process_conversation()
        
        # 等待记忆处理完成（最多等待10秒）
        import time
        start_time = time.time()
        while getattr(conversation_history, 'processing_memory', False) and (time.time() - start_time) < 10:
            time.sleep(0.1)
            
        # 关闭对话连接
        if conversation and hasattr(conversation, "close"):
            conversation.close()
        # 关闭播放器
        if 'b64_player' in globals() and b64_player:
            b64_player.shutdown()
    except:
        pass
    
    # 运行音频分析
    try:
        print('Analyzing recorded audio content... | 分析录音内容...')
        # 添加项目根目录到Python路径
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # 导入并运行音频分析
        from langas.audio_analyzer import AudioAnalyzer
        from config import DASHSCOPE_API_KEY
        
        analyzer = AudioAnalyzer(api_key=DASHSCOPE_API_KEY)
        report = analyzer.analyze_latest_recording()
        
        if "error" in report:
            print(f"❌ Audio analysis failed: {report['error']} | 音频分析失败: {report['error']}")
        else:
            print("✅ Audio analysis completed successfully! | 音频分析成功完成！")
            print("See detailed analysis report in the langrepo directory. | 详细分析报告请查看 langrepo 目录。")
            
            # 打印langrepo目录中的分析报告内容
            import glob
            import json
            # 查找最新的分析报告文件
            report_files = glob.glob("langrepo/*.json")
            if report_files:
                latest_report = max(report_files, key=os.path.getctime)
                print(f"\n📄 Latest analysis report: {latest_report}")
                try:
                    with open(latest_report, 'r', encoding='utf-8') as f:
                        report_data = json.load(f)
                    
                    # 打印声学分析结果
                    if "pronunciation" in report_data and "error" not in report_data["pronunciation"]:
                        print("\n=== Acoustic Analysis Results ===")
                        pronunciation = report_data["pronunciation"]
                        print(f"Duration: {pronunciation.get('duration', 'N/A'):.2f}s")
                        if "pitch" in pronunciation:
                            print(f"Pitch - Mean: {pronunciation['pitch'].get('mean', 'N/A'):.2f} Hz, "
                                  f"Min: {pronunciation['pitch'].get('min', 'N/A'):.2f} Hz, "
                                  f"Max: {pronunciation['pitch'].get('max', 'N/A'):.2f} Hz")
                        if "intensity" in pronunciation:
                            print(f"Intensity - Mean: {pronunciation['intensity'].get('mean', 'N/A'):.2f} dB, "
                                  f"Min: {pronunciation['intensity'].get('min', 'N/A'):.2f} dB, "
                                  f"Max: {pronunciation['intensity'].get('max', 'N/A'):.2f} dB")
                        if "formants" in pronunciation:
                            formants = pronunciation['formants']
                            print(f"Formants - F1: {formants.get('F1', 'N/A'):.2f} Hz, "
                                  f"F2: {formants.get('F2', 'N/A'):.2f} Hz, "
                                  f"F3: {formants.get('F3', 'N/A'):.2f} Hz, "
                                  f"F4: {formants.get('F4', 'N/A'):.2f} Hz")
                        
                        # 元音分析
                        if "vowels" in pronunciation and "error" not in pronunciation["vowels"]:
                            vowels = pronunciation["vowels"]
                            print(f"Vowels - F1 Mean: {vowels.get('F1_mean', 'N/A'):.2f} Hz, "
                                  f"F2 Mean: {vowels.get('F2_mean', 'N/A'):.2f} Hz")
                            if "comment" in vowels:
                                print(f"Vowel comments: {', '.join(vowels['comment']) if isinstance(vowels['comment'], list) else vowels['comment']}")
                        
                        # 辅音分析
                        if "consonants" in pronunciation and "error" not in pronunciation["consonants"]:
                            consonants = pronunciation["consonants"]
                            print(f"Consonants - Zero Crossing Rate: {consonants.get('zero_crossing_rate', 'N/A'):.4f}")
                            if "comment" in consonants:
                                print(f"Consonant comments: {consonants['comment']}")
                        
                        # 抖动和颤动
                        if "jitter_shimmer" in pronunciation and "error" not in pronunciation["jitter_shimmer"]:
                            js = pronunciation["jitter_shimmer"]
                            print(f"Jitter/Shimmer - Jitter: {js.get('jitter', 'N/A'):.4f}, "
                                  f"Shimmer: {js.get('shimmer', 'N/A'):.4f}")
                            if "comment" in js:
                                print(f"Voice quality comments: {', '.join(js['comment']) if isinstance(js['comment'], list) else js['comment']}")
                    
                    # AI内容分析
                    if "content_analysis" in report_data and "error" not in report_data["content_analysis"]:
                        print("\n=== AI Content Analysis ===")
                        content = report_data["content_analysis"].get("analysis", "N/A")
                        print(content)
                except Exception as e:
                    print(f"❌ Failed to read analysis report: {e}")
            else:
                print("No analysis report found in langrepo directory. | 在 langrepo 目录中未找到分析报告。")
                
            # 调用generate_intervention_prompt生成干预提示
            try:
                print("\nGenerating intervention prompt based on analysis results... | 根据分析结果生成干预提示...")
                # 添加项目根目录到Python路径（如果尚未添加）
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                
                # 导入并运行干预提示生成器
                from sysprompt_generate.generate_intervention_prompt import main as generate_intervention_prompt_main
                result = generate_intervention_prompt_main()
                
                if result:
                    print("✅ Intervention prompt generated successfully! | 干预提示生成成功！")
                else:
                    print("❌ Failed to generate intervention prompt | 生成干预提示失败")
            except Exception as e:
                print(f"❌ Failed to generate intervention prompt: {e} | 生成干预提示失败: {e}")
    except Exception as e:
        print(f"❌ Failed to run audio analysis: {e} | 运行音频分析失败: {e}")
    
    # 打印程序已停止信息
    print('\nQwen-Omni real-time conversation stopped. | Qwen-Omni 实时对话已停止。')
    # 退出程序
    sys.exit(0)


def main():
    """Main function | 主函数
    
    程序的主入口点，负责初始化和运行实时对话系统。
    """
    global conversation
    
    # 打印初始化信息
    print('Initializing Qwen-Omni real-time conversation system... | 正在初始化 Qwen-Omni 实时对话系统...')
    
    # Initialize conversation history with existing memory
    # 使用现有记忆初始化对话历史
    initialize_history_with_memory()
    
    # Set signal handler | 设置信号处理器
    # 设置信号处理器，用于处理Ctrl+C等信号
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create callback instance | 创建回调实例
    # 创建回调实例
    callback = MyCallback()
    
    # Create session | 创建会话
    # 创建实时对话会话
    conversation = OmniRealtimeConversation(
        model='qwen3-omni-flash-realtime',
        callback=callback,
        url=WEBSOCKET_URLS.get(REGION, WEBSOCKET_URLS['beijing'])
    )
    
    # Connect to server | 连接服务器
    # 连接到Qwen-Omni服务器
    try:
        conversation.connect()
    except TimeoutError as exc:
        logger.error("Failed to connect to realtime service within timeout: %s", exc)
        print("[Error] Unable to connect to Qwen-Omni real-time service within 5s. Please check network or firewall and retry.")
        return
    except Exception as exc:
        logger.error("Failed to connect to realtime service: %s", exc, exc_info=True)
        print(f"[Error] Unable to connect to Qwen-Omni real-time service: {exc}")
        return
    
    # Get system prompt with memory | 获取包含记忆的系统提示词
    # 获取包含记忆的系统提示词
    system_prompt = update_system_prompt_with_current_memory(conversation_history)
    
    # 打印系统提示词内容
    print("===== SYSTEM PROMPT START =====")
    print("Current system prompt:")
    print(system_prompt)
    print("===== SYSTEM PROMPT END =====")
    
    # Update session configuration | 更新会话配置
    # 更新会话配置，设置输出模式、语音等参数
    initial_session_kwargs = {
        "output_modalities": [MultiModality.AUDIO, MultiModality.TEXT],
        "voice": voice,
        "input_audio_format": getattr(AudioFormat, INPUT_AUDIO_FORMAT),
        "output_audio_format": getattr(AudioFormat, OUTPUT_AUDIO_FORMAT),
        "enable_input_audio_transcription": True,
        "input_audio_transcription_model": 'gummy-realtime-v1',
        "enable_turn_detection": True,
        "turn_detection_type": 'server_vad',
        "instructions": system_prompt,
    }

    conversation.update_session(**initial_session_kwargs)
    session_state.bind_conversation(conversation, initial_session_kwargs)
    
    # 打印连接成功信息
    print(f"Connected to Qwen-Omni service in {REGION} region | 已连接到 {REGION} 地域的 Qwen-Omni 服务")
    # 打印使用说明
    print("Press 'Ctrl+C' to stop conversation... | 按 'Ctrl+C' 停止对话...")
    
    # Main loop: Continuously read microphone data and send to server | 主循环：持续读取麦克风数据并发送到服务器
    # 主循环，持续读取麦克风数据并发送到服务器
    logger.info("Starting main conversation loop")
    while True:
        try:
            logger.debug("Main loop iteration")
            # 检查麦克风流是否存在
            if mic_stream:
                logger.debug("Checking if AI is playing voice")
                # Check if AI is playing voice, skip recording to avoid echo if so | 检查是否正在播放AI语音，如果是则跳过录音避免回音
                # 检查是否正在播放AI语音，如果是则跳过录音避免回音
                if hasattr(b64_player, 'is_currently_playing') and b64_player.is_currently_playing():
                    logger.debug("AI is speaking, skipping recording to avoid echo")
                    # AI is speaking, skip recording to avoid echo | AI正在说话，跳过录音以避免回音
                    # AI正在说话，跳过录音以避免回音
                    time.sleep(0.01)  # Brief sleep to avoid high CPU usage | 短暂休眠避免CPU占用过高
                    # 短暂休眠避免CPU占用过高
                    continue
                    
                logger.debug("Reading audio data from microphone")
                # 从麦克风读取音频数据
                audio_data = mic_stream.read(3200, exception_on_overflow=False)
                logger.debug(f"Sending audio data to server, data length: {len(audio_data)}")
                # 将音频数据编码为Base64格式
                audio_b64 = base64.b64encode(audio_data).decode('ascii')
                # 将音频数据发送到服务器
                conversation.append_audio(audio_b64)
            else:
                logger.warning("Microphone stream is None, breaking loop")
                # 如果麦克风流不存在，则退出循环
                break
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, breaking loop")
            # 捕获键盘中断异常（Ctrl+C）
            break
        except Exception as e:
            logger.error(f"[Error] Microphone read failed: {e} | [错误] 麦克风读取失败: {e}", exc_info=True)
            # 打印错误信息
            print(f"[Error] Microphone read failed: {e} | [错误] 麦克风读取失败: {e}")
            # 发生异常时退出循环
            break


# 程序入口点
if __name__ == '__main__':
    # 调用主函数
    main()