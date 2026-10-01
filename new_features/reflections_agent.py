"""
Reflections Agent: Manage conversation segments by deduplicating, compressing,
and synthesizing higher-level reflections using a LangGraph create_agent.

- Reads existing memory from memory_system_prompt.py (profile + segments)
- Accepts example messages (inline demo) and existing segments
- Produces an updated segment list (deduped + compressed + reflections)
- Writes back to memory_system_prompt.py using the same formatting helpers

Note:
- We reuse _build_langchain_chat_model from mem_bank.py to keep model config consistent
- We avoid modifying mem_bank.py. This script is standalone and safe to run
- For simplicity, we provide all memories directly (no external vector store)
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mem_bank import (
    _build_langchain_chat_model,  # reuse model settings
    load_memory_from_system_prompt_file,
)

# Ensure .env is loaded so QWEN credentials are available in this standalone script
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# We replicate minimal writers locally to avoid mutating mem_bank.py
MEMORY_SYSTEM_PROMPT_FILE = Path(__file__).resolve().parent / "memory_system_prompt.py"

PROFILE_KEY_MAP = {
    "Name": "名字",
    "Timezone": "时区",
    "Likes": "喜欢",
    "Dislikes": "不喜欢",
    "Emotional Needs": "情感需求",
    "Communication Style": "沟通风格",
}


def _format_profile_section(profile: Dict[str, Any]) -> List[str]:
    if not profile:
        return ["- 暂无记录"]

    lines: List[str] = []
    for key, label in PROFILE_KEY_MAP.items():
        value = profile.get(key)
        if not value:
            continue
        if isinstance(value, list):
            pretty = "，".join(value)
        else:
            pretty = str(value)
        lines.append(f"- {label}: {pretty}")
    if not lines:
        lines.append("- 暂无记录")
    return lines


def _format_segments_section(segments: List[Dict[str, Any]]) -> List[str]:
    if not segments:
        return ["暂无对话片段。"]

    lines: List[str] = []
    for seg in segments:
        summary = (seg.get("summary") or "").strip()
        convr_num = seg.get("convr_num")
        convr_index = seg.get("convr_index")
        timestamp = seg.get("timestamp", "")
        if not summary:
            continue
        if convr_num and convr_index:
            if timestamp:
                tag = f"CONVR{convr_num}-{convr_index}[{timestamp}]"
            else:
                tag = f"CONVR{convr_num}-{convr_index}"
        else:
            tag = "CONVR?-?"
        lines.append(f"{tag}. {summary}")
    return lines or ["暂无对话片段。"]


def _build_system_prompt_string(content: Dict[str, Any]) -> str:
    from datetime import datetime, timezone

    profile = content.get("User Profile", {}) or {}
    segments = content.get("Previous Conversation Segments", []) or []
    user_timezone = profile.get("Timezone", "Unknown")

    # Current UTC time
    utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: List[str] = []
    lines.append("个人设定你叫灵灵.回答简短，回答不要有那么多的符号或者表情，你是一个心理咨询师，回答要有温度和情感。")
    lines.append("")
    lines.append(f"当前时间: {utc_now} (用户时区: {user_timezone}) 请把这个时间当作 NOW")
    lines.append("")
    lines.append("关于用户的信息：")
    lines.extend(_format_profile_section(profile))
    lines.append("")
    lines.append("对话历史片段：")
    lines.extend(_format_segments_section(segments))

    return "\n".join(lines)


def _write_system_prompt_file(filepath: Path, prompt_content: str) -> None:
    escaped = prompt_content.replace("\\", "\\\\").replace("\"", "\\\"")
    python_code = (
        '"""\nMemory System Prompt 管理模块\n用于存储经过对话学到的用户信息和对话片段\n这个文件由 mem_bank/reflections_agent 自动更新\n"""\n\n'
        f'MEMORY_SYSTEM_PROMPT = """{escaped}"""\n'
    )
    filepath.write_text(python_code, encoding="utf-8")


# -------------------- Agent scaffolding --------------------

from langchain.agents import create_agent


def _dedupe_by_text(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Simple exact-text dedupe before sending to the LLM."""
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for s in segments:
        text = (s.get("summary") or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    return deduped


def _build_reflection_system_prompt(profile: Dict[str, Any], segments: List[Dict[str, Any]]) -> str:
    """Instruct the agent to output JSON with refined segments and reflections."""
    profile_lines = "\n".join(_format_profile_section(profile))
    seg_lines = "\n".join(_format_segments_section(segments))
    return (
        "You are a memory manager for conversation segments.\n"
        "Goals:\n"
        "1) Remove duplicates and near-duplicates.\n"
        "2) Merge overlapping items and compress them into concise summaries (< 30 tokens each).\n"
        "3) Add 1-3 higher-level REFLECTIONS that synthesize cross-cutting insights.\n\n"
        "Output STRICT JSON only with keys: {\"segments\": [ {\"summary\": str }... ], \"reflections\": [str, ... ]}.\n"
        "- Do not include any extra commentary or markdown.\n"
        "- Keep segments in chronological sense if clear, otherwise best effort.\n"
        "- Segments should be directly usable in a retrieval-augmented system.\n\n"
        f"User Profile (reference):\n{profile_lines}\n\n"
        f"Existing Segments:\n{seg_lines}\n\n"
        "Now, produce the JSON result."
    )


def _invoke_reflection_json(chat_model, agent, system_prompt: str, example_messages: List[Dict[str, str]]) -> str:
    """Call the agent (or model fallback) and return raw assistant content string."""
    messages = [{"role": "system", "content": system_prompt}] + example_messages

    # Try agent first
    try:
        result = agent.invoke({"messages": messages})  # type: ignore[arg-type]
    except Exception:
        result = None

    output_text: Optional[str] = None
    if isinstance(result, dict):
        msgs = result.get("messages")
        if isinstance(msgs, list) and msgs:
            for m in reversed(msgs):
                if isinstance(m, dict):
                    role = m.get("role")
                    content = m.get("content")
                    if role in ("assistant", "ai") and isinstance(content, str) and content.strip():
                        output_text = content
                        break
                else:
                    role = getattr(m, "type", None) or getattr(m, "role", None)
                    content = getattr(m, "content", None)
                    if role in ("assistant", "ai") and isinstance(content, str) and content.strip():
                        output_text = content
                        break
        if output_text is None:
            maybe = result.get("content") or result.get("output") or result.get("final")
            if isinstance(maybe, str) and maybe.strip():
                output_text = maybe

    # Fallback to direct model call
    if not output_text:
        direct = chat_model.invoke(messages)  # type: ignore[arg-type]
        if hasattr(direct, "content"):
            output_text = getattr(direct, "content")
        elif isinstance(direct, dict):
            output_text = direct.get("content")

    if not output_text or not isinstance(output_text, str):
        raise RuntimeError("Agent did not return content to parse.")

    return output_text


def _parse_reflection_output(output_text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Parse the strict JSON output into (segments, reflections)."""
    json_str = output_text.strip()
    if json_str.startswith("```"):
        import re as _re
        json_str = _re.sub(r"^```(json)?|```$", "", json_str, flags=_re.MULTILINE).strip()
    payload = json.loads(json_str)

    segments_out: List[Dict[str, Any]] = []
    for item in payload.get("segments", []) or []:
        if isinstance(item, dict):
            summary = (item.get("summary") or "").strip()
        else:
            summary = str(item).strip()
        if not summary:
            continue
        segments_out.append({"summary": summary})

    reflections_out: List[str] = []
    for ref in payload.get("reflections", []) or []:
        text = str(ref).strip()
        if text:
            reflections_out.append(text)

    return segments_out, reflections_out


def _critique_and_feedback(chat_model, current_json_text: str) -> Optional[str]:
    """Ask the model to self-critique the JSON and return brief guidance or None if pass."""
    critique_prompt = (
        "You are a strict reviewer of the following JSON output that summarises conversation segments and reflections.\n"
        "Criteria: no duplicates, clear concise segments (<30 tokens), chronological where possible, reflections are higher-level and non-redundant (verify these will be useful for emotional support), JSON valid.\n"
        "Respond with strict JSON: {\"pass\": true|false, \"comment\": string, \"guidance\": string}. Keep comments <= 60 tokens.\n\n"
        f"Output to review:\n{current_json_text}\n"
    )
    resp = chat_model.invoke([{"role": "user", "content": critique_prompt}])  # type: ignore[arg-type]
    content = getattr(resp, "content", None) if hasattr(resp, "content") else (resp.get("content") if isinstance(resp, dict) else None)
    if not isinstance(content, str):
        return None
    try:
        review = json.loads(content.strip().strip("`"))
        if isinstance(review, dict) and review.get("pass") is True:
            return None
        guidance = review.get("guidance") or review.get("comment")
        return str(guidance) if guidance else None
    except Exception:
        return None


def run_reflection(existing_profile: Dict[str, Any], existing_segments: List[Dict[str, Any]], example_messages: List[Dict[str, str]], iterations: int = 3) -> Tuple[List[Dict[str, Any]], List[str]]:
    chat_model = _build_langchain_chat_model()

    # Build agent with no tools; we pass context through the system message
    agent = create_agent(
        model=chat_model,
        tools=[],
    )

    base_segments = _dedupe_by_text(existing_segments)
    guidance_note: Optional[str] = None
    last_good_segments: List[Dict[str, Any]] = []
    last_good_reflections: List[str] = []

    for i in range(max(1, min(iterations, 3))):
        system_prompt = _build_reflection_system_prompt(existing_profile, base_segments)
        if guidance_note:
            system_prompt += f"\n\nCritique Guidance: {guidance_note}\nPlease revise the JSON accordingly."

        output_text = _invoke_reflection_json(chat_model, agent, system_prompt, example_messages)

        try:
            segments_out, reflections_out = _parse_reflection_output(output_text)
            # Save as last good
            last_good_segments, last_good_reflections = segments_out, reflections_out
        except Exception:
            # Could not parse; attempt critique anyway with raw
            segments_out, reflections_out = [], []

        # Ask for critique; if passes, break early
        guidance = _critique_and_feedback(chat_model, output_text)
        if not guidance:
            break
        guidance_note = guidance

    return last_good_segments, last_good_reflections


def _merge_segments_for_prompt(old_segments: List[Dict[str, Any]], new_segments: List[Dict[str, Any]], reflections: List[str]) -> List[Dict[str, Any]]:
    """
    Merge refined segments back for system prompt writing.
    - Preserve existing convr_num/convr_index where the text matches exactly
    - For new items (including reflections), keep no convr fields so they render as CONVR?-?
    """
    index_by_text = { (s.get("summary") or "").strip().lower(): s for s in old_segments }

    merged: List[Dict[str, Any]] = []

    # Add refined segments
    seen = set()
    for s in new_segments:
        text = (s.get("summary") or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        if key in index_by_text:
            orig = index_by_text[key]
            merged.append({
                "summary": text,
                "convr_num": orig.get("convr_num"),
                "convr_index": orig.get("convr_index"),
                **({"timestamp": orig.get("timestamp")} if orig.get("timestamp") else {}),
            })
        else:
            merged.append({"summary": text})

    # Append explicit reflections as additional segments (tagged)
    for ref in reflections:
        tag_text = f"REFLECTION: {ref}"
        if tag_text.lower() in seen:
            continue
        merged.append({"summary": tag_text})

    return merged


def demo_run() -> None:
    """Run a demo reflection pass using inline example messages."""
    memory = load_memory_from_system_prompt_file(MEMORY_SYSTEM_PROMPT_FILE)
    profile = memory.get("User Profile", {}) or {}
    segments = memory.get("Previous Conversation Segments", []) or []

    # Example conversation messages (minimal demo)
    example_messages: List[Dict[str, str]] = [
        {"role": "user", "content": "Hi, I’m Alex. I’ve been feeling anxious about work deadlines."},
        {"role": "assistant", "content": "Thanks for sharing, Alex. What tends to trigger the anxiety most?"},
        {"role": "user", "content": "Unclear priorities. I also try running on weekends to decompress."},
        {"role": "assistant", "content": "Got it. Let’s explore prioritization strategies and routines that help."},
        {"role": "user", "content": "I’d like brief checklists and reminders. I dislike cluttered suggestions."},
    ]

    new_segments, reflections = run_reflection(profile, segments, example_messages)

    merged_segments = _merge_segments_for_prompt(segments, new_segments, reflections)

    payload = {
        "User Profile": profile,
        "Previous Conversation Segments": merged_segments,
    }

    prompt = _build_system_prompt_string(payload)
    _write_system_prompt_file(MEMORY_SYSTEM_PROMPT_FILE, prompt)

    print("Updated segments (count=", len(merged_segments), ")\n---")
    for s in merged_segments:
        print("-", s.get("summary"))


if __name__ == "__main__":
    demo_run()
