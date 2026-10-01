{
  "analysis_prompt": "请分析发音、词汇、语法和语调，并给建议"
}
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
音频分析模块，用于分析录音文件的发音、词义和语法问题
Audio analysis module for analyzing pronunciation, lexical and grammatical issues in recordings
"""

import os
import json
import glob
import logging
import statistics
from typing import Dict, Any, Optional

# ---------------- Praat Setup ----------------
try:
    import parselmouth
    from parselmouth import Sound
    from parselmouth.praat import call
    PRAAT_AVAILABLE = True
except ImportError:
    PRAAT_AVAILABLE = False
    print("⚠️ Warning: praat-parselmouth missing. Install via:")
    print("   pip install praat-parselmouth")

# ---------------- Qwen Setup ----------------
try:
    import dashscope
    from dashscope import MultiModalConversation
    DASHSCOPE_AVAILABLE = True
except:
    DASHSCOPE_AVAILABLE = False

# ---------------- Logger ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# ✨ Audio Analyzer Class
# ============================================================

class AudioAnalyzer:
    def __init__(self, api_key: Optional[str] = None, prompt_file: str = "config/analysis_prompt.json"):
        self.api_key = api_key
        if api_key and DASHSCOPE_AVAILABLE:
            dashscope.api_key = api_key

        # Load analysis prompt from JSON file
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.analysis_prompt = config.get("analysis_prompt", "请分析发音、词汇、语法和语调，并给建议")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load prompt from {prompt_file}, using default: {e}")
            self.analysis_prompt = "请分析发音、词汇、语法和语调，并给建议"

        os.makedirs("langrepo", exist_ok=True)

    # --------------------------------------------
    # 🎤 Pronunciation Analysis via Praat
    # --------------------------------------------
    def analyze_pronunciation(self, file: str) -> Dict[str, Any]:
        if not PRAAT_AVAILABLE:
            return {"error": "Praat not installed"}
        try:
            snd = Sound(file)
            duration = snd.get_total_duration()

            # Pitch
            pitch = call(snd, "To Pitch", 0.0, 75, 600)
            mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")
            min_f0 = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
            max_f0 = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")

            # Intensity
            intensity = call(snd, "To Intensity", 75, 0)
            mean_intensity = call(intensity, "Get mean", 0, 0)
            min_intensity = call(intensity, "Get minimum", 0, 0, "Parabolic")
            max_intensity = call(intensity, "Get maximum", 0, 0, "Parabolic")

            # Formants
            formant = call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
            f1 = call(formant, "Get mean", 1, 0, 0, "Hertz")
            f2 = call(formant, "Get mean", 2, 0, 0, "Hertz")
            f3 = call(formant, "Get mean", 3, 0, 0, "Hertz")
            f4 = call(formant, "Get mean", 4, 0, 0, "Hertz")

            vowels = self._analyze_vowels(snd, formant)
            consonants = self._analyze_consonants(snd)

            jitter_shimmer = self._jitter_shimmer(snd)

            return {
                "duration": duration,
                "pitch": {"mean": mean_f0, "min": min_f0, "max": max_f0},
                "intensity": {"mean": mean_intensity, "min": min_intensity, "max": max_intensity},
                "formants": {"F1": f1, "F2": f2, "F3": f3, "F4": f4},
                "vowels": vowels,
                "consonants": consonants,
                "jitter_shimmer": jitter_shimmer
            }
        except Exception as e:
            return {"error": str(e)}

    # --------------------------------------------
    # 🔤 Vowel Clarity
    # --------------------------------------------
    def _analyze_vowels(self, snd, formant):
        f1_values, f2_values = [], []
        step = 0.01
        t = 0
        duration = snd.get_total_duration()
        while t < duration:
            try:
                f1 = formant.get_value_at_time(1, t)
                f2 = formant.get_value_at_time(2, t)
                # 检查值是否有效（不是None也不是NaN）
                if f1 is not None and f2 is not None and f1 == f1 and f2 == f2:  # f == f 检查NaN
                    f1_values.append(f1)
                    f2_values.append(f2)
            except:
                pass
            t += step

        if not f1_values or not f2_values:
            return {"error": "no vowel data"}

        import statistics
        mf1 = statistics.mean(f1_values)
        mf2 = statistics.mean(f2_values)

        issues=[]
        if not (200 < mf1 < 1000): issues.append("Tongue height abnormal (F1)")
        if not (800 < mf2 < 2500): issues.append("Tongue front-back abnormal (F2)")

        return {
            "F1_mean": mf1,
            "F2_mean": mf2,
            "comment": issues or ["Vowels clear & normal"]
        }

    # --------------------------------------------
    # 🫁 Consonant Energy + ZCR
    # --------------------------------------------
    def _analyze_consonants(self, snd):
        try:
            samples = snd.values[0]
            frame = 160
            zcr = []
            for i in range(0, len(samples) - frame, frame):
                frame_vals = samples[i:i + frame]
                zero_cross = sum(1 for j in range(1, len(frame_vals)) 
                                if frame_vals[j] * frame_vals[j-1] < 0)
                zcr.append(zero_cross / len(frame_vals))

            mean_zcr = statistics.mean(zcr) if zcr else 0

            return {
                "zero_crossing_rate": mean_zcr,
                "comment": "Likely voiced speech" if mean_zcr < 0.1 else "Noisy / many consonants"
            }
        except Exception as e:
            return {"error": f"consonant analysis failed: {str(e)}"}

    # --------------------------------------------
    # 🎚️ Jitter & Shimmer
    # --------------------------------------------
    def _jitter_shimmer(self, snd):
        try:
            pp = call(snd, "To PointProcess (periodic, cc)", 75, 600)
            jitter = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

            comment = []
            if jitter > 0.01: comment.append("Voice instability")
            if shimmer > 0.035: comment.append("Amplitude instability")
            return {"jitter": jitter, "shimmer": shimmer, "comment": comment or ["Stable voice"]}
        except Exception as e:
            return {"error": f"jitter/shimmer failed: {str(e)}"}

    # --------------------------------------------
    # 🤖 Qwen Audio Analysis
    # --------------------------------------------
    def analyze_qwen(self, file: str, acoustic_analysis: Dict[str, Any] = None):
        if not DASHSCOPE_AVAILABLE or not self.api_key:
            return {"error": "Qwen not available"}
        try:
            # 从JSON文件加载分析提示文本
            with open(os.path.join(os.path.dirname(__file__), 'prompts.json'), 'r', encoding='utf-8') as f:
                prompts = json.load(f)
            base_prompt = prompts.get('audio_analysis_prompt', '你是一个语言发育分析专家')

            # 如果提供了声学分析结果，则构造包含声学分析的完整提示
            if acoustic_analysis:
                # 清理声学分析数据，移除错误信息
                clean_analysis = {k: v for k, v in acoustic_analysis.items() if k != 'error'}
                if clean_analysis:
                    # 构造完整的提示文本，包含声学分析结果
                    import json as json_module
                    full_prompt = f"{base_prompt}\n\n声学分析结果：{json_module.dumps(clean_analysis, ensure_ascii=False, indent=2)}"
                else:
                    full_prompt = base_prompt
            else:
                full_prompt = base_prompt
            
            r = MultiModalConversation.call(
                model="qwen-audio-turbo-latest",
                messages=[{
                    "role": "user",
                    "content":[{"audio": file}, {"text": full_prompt}]
                }],
                result_format="message",
            )
            content = r["output"]["choices"][0]["message"]["content"]
            return {"analysis": content[0]["text"] if isinstance(content,list) else content}
        except Exception as e:
            return {"error": str(e)}

    # --------------------------------------------
    # 🕵️ Analyze latest recording
    # --------------------------------------------
    def analyze_latest_recording(self):
        files = glob.glob("recordings/*.wav")
        if not files:
            return {"error": "No wav found in recordings/"}
        latest = max(files, key=os.path.getctime)
        logger.info(f"🎧 Analyzing: {latest}")

        # 先进行声学分析
        pronunciation_analysis = self.analyze_pronunciation(latest)
        
        # 将声学分析结果传递给Qwen分析
        content_analysis = self.analyze_qwen(latest, pronunciation_analysis)

        report = {
            "audio_file": latest,
            "pronunciation": pronunciation_analysis,
            "content_analysis": content_analysis,
        }

        import datetime
        report["analysis_time"] = datetime.datetime.now().isoformat()

        self.save_report(report)
        return report

    # --------------------------------------------
    # 💾 Save JSON
    # --------------------------------------------
    def save_report(self, report):
        fname = f"langrepo/report_{report['analysis_time'].replace(':','-')}.json"
        with open(fname,"w",encoding="utf-8") as f:
            json.dump(report,f,ensure_ascii=False,indent=2)
        print(f"✅ Report saved:", fname)

# ============================================================
#  🚀 Main
# ============================================================

def main():
    import sys
    import os
    # 添加项目根目录到路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from config import DASHSCOPE_API_KEY

    analyzer = AudioAnalyzer(api_key=DASHSCOPE_API_KEY)
    report = analyzer.analyze_latest_recording()
    
    if "error" in report:
        print(f"❌ Analysis failed: {report['error']}")
        return

    print("\n=== Summary ===")
    print(f"File: {report['audio_file']}")
    if "error" not in report["pronunciation"]:
        pronunciation = report["pronunciation"]
        print(f"Duration: {pronunciation.get('duration', 'N/A'):.2f}s")
        print(f"Mean pitch: {pronunciation.get('pitch', {}).get('mean', 'N/A'):.2f} Hz")
        print(f"Mean intensity: {pronunciation.get('intensity', {}).get('mean', 'N/A'):.2f} dB")
        
        # Formants
        formants = pronunciation.get('formants', {})
        if formants:
            print(f"Formants - F1: {formants.get('F1', 'N/A'):.2f} Hz, F2: {formants.get('F2', 'N/A'):.2f} Hz, F3: {formants.get('F3', 'N/A'):.2f} Hz, F4: {formants.get('F4', 'N/A'):.2f} Hz")
        
        # Vowels
        vowels = pronunciation.get('vowels', {})
        if vowels and "error" not in vowels:
            print(f"Vowels - F1: {vowels.get('F1_mean', 'N/A'):.2f} Hz, F2: {vowels.get('F2_mean', 'N/A'):.2f} Hz")
            if 'comment' in vowels:
                print(f"Vowel issues: {', '.join(vowels['comment']) if isinstance(vowels['comment'], list) else vowels['comment']}")
        
        # Consonants
        consonants = pronunciation.get('consonants', {})
        if consonants and "error" not in consonants:
            print(f"Consonants - Zero Crossing Rate: {consonants.get('zero_crossing_rate', 'N/A'):.4f}")
            if 'comment' in consonants:
                print(f"Consonant analysis: {consonants['comment']}")
        
        # Jitter and Shimmer
        jitter_shimmer = pronunciation.get('jitter_shimmer', {})
        if jitter_shimmer and "error" not in jitter_shimmer:
            print(f"Jitter: {jitter_shimmer.get('jitter', 'N/A'):.4f}, Shimmer: {jitter_shimmer.get('shimmer', 'N/A'):.4f}")
            if 'comment' in jitter_shimmer:
                print(f"Voice quality: {', '.join(jitter_shimmer['comment']) if isinstance(jitter_shimmer['comment'], list) else jitter_shimmer['comment']}")
    
    # AI Content Analysis
    if "error" not in report["content_analysis"]:
        print(f"\n=== AI Content Analysis ===")
        content = report["content_analysis"].get("analysis", "N/A")
        print(content)
    elif "error" in report["content_analysis"]:
        print(f"\n❌ AI Content Analysis failed: {report['content_analysis']['error']}")

if __name__ == "__main__":
    main()
