"""Unified memory management for the Qwen-Omni conversation system."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

# =============================================================================
# Logging configuration
# =============================================================================

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("memory_processing.log", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )


# =============================================================================
# Constants & Paths
# =============================================================================

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MEMORY_SYSTEM_PROMPT_FILE = ROOT / "memory_system_prompt.py"
CONVERSATION_COUNTER_FILE = ROOT / ".conversation_counter"
DEFAULT_BACKEND: Literal["langmem", "qwen"] = "langmem"


# =============================================================================
# Data models (shared by both backends)
# =============================================================================


class UserProfile(BaseModel):
    """Represents the full representation of a user."""

    name: Optional[str] = Field(default=None, alias="Name")
    timezone: Optional[str] = Field(default=None, alias="Timezone")
    likes: List[str] = Field(default_factory=list, alias="Likes")
    dislikes: List[str] = Field(default_factory=list, alias="Dislikes")
    emotional_needs: List[str] = Field(default_factory=list, alias="Emotional Needs")
    communication_style: Optional[str] = Field(default=None, alias="Communication Style")

    model_config = ConfigDict(populate_by_name=True)


class SegmentSummary(BaseModel):
    """A summary of a part of the conversation."""

    summary: str
    convr_num: Optional[int] = Field(default=None, description="Conversation number this segment is from")
    convr_index: Optional[int] = Field(default=None, description="Sequential index within the conversation")


class SegmentCollection(BaseModel):
    """Container for multiple topical segment summaries."""

    segments: List[SegmentSummary] = Field(default_factory=list)


# =============================================================================
# Utility helpers
# =============================================================================


def _ensure_counter_file() -> None:
    if not CONVERSATION_COUNTER_FILE.exists():
        CONVERSATION_COUNTER_FILE.write_text("0", encoding="utf-8")


def _read_counter() -> int:
    _ensure_counter_file()
    try:
        return int(CONVERSATION_COUNTER_FILE.read_text(encoding="utf-8").strip() or "0")
    except ValueError:
        logger.warning("Conversation counter file corrupted. Resetting to 0.")
        CONVERSATION_COUNTER_FILE.write_text("0", encoding="utf-8")
        return 0


def _increment_counter() -> int:
    current = _read_counter() + 1
    CONVERSATION_COUNTER_FILE.write_text(str(current), encoding="utf-8")
    return current


def peek_current_conversation_number() -> int:
    """Return the latest recorded conversation number without incrementing."""

    return _read_counter()


def reset_conversation_counter(value: int = 0) -> None:
    """Reset the global conversation counter (useful for tests)."""

    if value < 0:
        raise ValueError("Conversation counter cannot be negative")
    CONVERSATION_COUNTER_FILE.write_text(str(value), encoding="utf-8")


def _prepare_chat_messages(messages: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    chat: List[Dict[str, str]] = []
    for item in messages:
        role = item.get("role")
        content = item.get("content", "")
        if role in {"user", "assistant"} and content:
            chat.append({"role": role, "content": content})
    return chat


def _merge_list_values(existing: List[str], incoming: Iterable[str]) -> List[str]:
    merged = list(existing)
    for value in incoming:
        if isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                continue
            if candidate not in merged:
                merged.append(candidate)
            continue

        if value not in merged:
            merged.append(value)
    return merged


def _unwrap_langmem_payload(payload: Any, expected_types: Tuple[type, ...]) -> Any:
    """Unwrap langmem ExtractedMemory objects to their underlying value."""

    max_depth = 5
    depth = 0
    current = payload
    while depth < max_depth:
        if isinstance(current, expected_types):
            return current
        if hasattr(current, "value"):
            next_value = getattr(current, "value")
            if next_value is not None:
                current = next_value
                depth += 1
                continue
        if hasattr(current, "content"):
            content = getattr(current, "content")
            if isinstance(content, str):
                try:
                    current = json.loads(content)
                except json.JSONDecodeError:
                    current = content
            elif content is not None:
                current = content
            else:
                current = getattr(current, "_asdict", lambda: {})().get("content")
            depth += 1
            continue
        if hasattr(current, "_asdict"):
            as_dict = current._asdict()
            content_value = as_dict.get("content") if isinstance(as_dict, dict) else None
            if content_value is not None:
                current = content_value
            else:
                current = as_dict
            depth += 1
            continue
        break
    return current


# =============================================================================
# Backend 1: langmem
# =============================================================================


def _build_langchain_chat_model():  # pragma: no cover - heavy external dependency
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("QWEN_API_KEY (or DASHSCOPE_API_KEY) must be set for langmem backend")

    base_url = os.getenv("QWEN_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL")
    model_name = os.getenv("QWEN_MODEL", "qwen-plus")

    return ChatOpenAI(  # type: ignore[call-arg]
        api_key=SecretStr(api_key.strip()),
        base_url=base_url.strip() if base_url else None,
        model=model_name.strip(),
        temperature=0.0,
        streaming=False,
    )


def _extract_profile_langmem(
    chat_messages: List[Dict[str, Any]],
    existing_profile: Dict[str, Any]
) -> Dict[str, Any]:  # pragma: no cover - relies on external service
    from langmem import create_memory_manager

    chat_model = _build_langchain_chat_model()

    profile_manager = create_memory_manager(
        chat_model,
        schemas=[UserProfile],
        enable_inserts=True,
        enable_updates=True,
        instructions=(
            "Extract durable user profile facts from the dialogue. Capture only information the user states as ongoing or "
            "habitual (e.g., stable preferences, identity details, long-term emotional tendencies). Ignore fleeting moods "
            "unless the user frames them as recurring. IMPORTANT: Update and extend existing profile information rather than "
            "replacing it."
        ),
    )

    existing_profile_model = None
    if existing_profile:
        try:
            existing_profile_model = UserProfile.model_validate(existing_profile)
        except ValidationError as exc:
            logger.warning("Existing profile validation failed for langmem backend: %s", exc)

    existing_memories = [existing_profile_model] if existing_profile_model else []

    profile_updates_result = profile_manager.invoke(  # type: ignore[arg-type]
        {"messages": chat_messages}, existing_memories=existing_memories
    )

    if profile_updates_result and not isinstance(profile_updates_result, list):
        profile_updates_result = [profile_updates_result]

    normalized_updates: List[UserProfile] = []
    for idx, update in enumerate(profile_updates_result or []):
        payload = _unwrap_langmem_payload(update, (dict, UserProfile))
        if isinstance(payload, UserProfile):
            update_model = payload
        elif isinstance(payload, dict):
            try:
                update_model = UserProfile.model_validate(payload)
            except ValidationError as exc:
                logger.warning("Discarding invalid profile update from langmem: %s", exc)
                continue
        else:
            try:
                update_model = UserProfile.model_validate(payload)
            except ValidationError as exc:
                logger.warning("Discarding invalid profile update from langmem: %s", exc)
                continue
        normalized_updates.append(update_model)

    merged_profile: Dict[str, Any] = deepcopy(existing_profile) if existing_profile else {}

    for update_model in normalized_updates:
        update_dict = update_model.model_dump(by_alias=True, exclude_none=True)
        for key, value in update_dict.items():
            if key in {"Likes", "Dislikes", "Emotional Needs"}:
                merged_profile[key] = _merge_list_values(merged_profile.get(key, []), value)
            else:
                merged_profile[key] = value

    return merged_profile


def _extract_segments_langmem(
    chat_messages: List[Dict[str, Any]],
    existing_segments: List[Dict[str, Any]],
    conversation_num: int,
    timestamp: str = ""
) -> List[Dict[str, Any]]:  # pragma: no cover - relies on external service
    from langmem import create_memory_manager

    chat_model = _build_langchain_chat_model()

    seg_instructions = (
        "Read the conversation chronologically and emit one segment each time the discussion shifts to a new topic or subtask. "
        "For every segment, include both user statements and assistant proposals that belong to that topic. Keep each summary "
        "under 30 tokens, pack all related facts together (e.g., travel details across multiple cities), and capture emotional "
        "updates as their own segments when they describe the user's state. Do not mix unrelated topics and preserve chronological ordering."
    )

    segment_manager = create_memory_manager(
        chat_model,
        schemas=[SegmentCollection],
        enable_inserts=True,
        enable_updates=True,
        enable_deletes=True,
        instructions=seg_instructions,
    )

    extracted_segments_result = segment_manager.invoke({"messages": chat_messages})  # type: ignore[arg-type]

    if extracted_segments_result and not isinstance(extracted_segments_result, list):
        extracted_segments_result = [extracted_segments_result]

    merged_segments = deepcopy(existing_segments) if existing_segments else []
    existing_summaries = {
        (seg.get("summary", "") if isinstance(seg, dict) else "").strip()
        for seg in merged_segments
    }

    for idx, item in enumerate(extracted_segments_result or []):
        payload = _unwrap_langmem_payload(item, (dict, SegmentCollection))
        try:
            collection = SegmentCollection.model_validate(payload)
        except ValidationError as exc:
            logger.warning("Discarding invalid segment collection from langmem: %s", exc)
            continue

        for seg in collection.segments:
            summary = seg.summary.strip()
            if not summary:
                continue
            if summary in existing_summaries:
                continue

            next_index = (
                sum(1 for existing in merged_segments if existing.get("convr_num") == conversation_num) + 1
            )
            segment_entry: Dict[str, Any] = {
                "summary": summary,
                "convr_num": conversation_num,
                "convr_index": next_index,
            }
            # Add timestamp only for the first segment of this conversation
            if next_index == 1 and timestamp:
                segment_entry["timestamp"] = timestamp
            merged_segments.append(segment_entry)
            existing_summaries.add(summary)

    return merged_segments


# =============================================================================
# Backend 2: QWEN function calling
# =============================================================================


class QwenClient:  # pragma: no cover - relies on external service
    """Wrapper for QWEN chat completion calls with function schema support."""

    def __init__(self) -> None:
        from openai import OpenAI

        api_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("QWEN_API_KEY (or DASHSCOPE_API_KEY) must be set for QWEN function calling backend")

        base_url = os.getenv("QWEN_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.model_name = os.getenv("QWEN_MODEL", "qwen-plus")
        self.client = OpenAI(api_key=api_key.strip(), base_url=base_url.strip())

    def call_function(
        self,
        messages: List[Dict[str, str]],
        function_schema: Dict[str, Any],
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        messages_param: Any = messages
        functions_param: Any = [function_schema]
        response = self.client.chat.completions.create(  # type: ignore
            model=self.model_name,
            messages=messages_param,
            functions=functions_param,
            function_call={"name": function_schema["name"]},
            temperature=temperature,
        )

        choice = response.choices[0]
        function_call = getattr(choice.message, "function_call", None)
        if not function_call or not function_call.arguments:
            logger.warning("QWEN function call returned no arguments. Raw response: %s", response)
            return {}

        try:
            return json.loads(function_call.arguments)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse QWEN function call payload: %s", exc)
            return {}


def _extract_profile_qwen(
    chat_messages: List[Dict[str, str]],
    existing_profile: Dict[str, Any]
) -> Dict[str, Any]:  # pragma: no cover - relies on external service
    existing_profile_json = json.dumps(existing_profile or {}, ensure_ascii=False)
    system_prompt = (
        "Analyse the dialogue to update the long-term user profile. Capture only facts the user presents as ongoing or habitual "
        "(identity details, enduring preferences, repeated emotional tendencies). Ignore one-off moods unless the user frames "
        "the state as recurring. Use the tool output to report new or reinforced information and avoid contradicting existing data. "
        f"Existing profile JSON: {existing_profile_json}."
    )

    messages = [{"role": "system", "content": system_prompt}] + chat_messages

    function_schema = {
        "name": "update_user_profile",
        "description": "Store verified user profile attributes mentioned in the conversation.",
        "parameters": {
            "type": "object",
            "properties": {
                "Name": {"type": "string"},
                "Timezone": {"type": "string"},
                "Likes": {"type": "array", "items": {"type": "string"}},
                "Dislikes": {"type": "array", "items": {"type": "string"}},
                "Emotional Needs": {"type": "array", "items": {"type": "string"}},
                "Communication Style": {"type": "string"},
            },
        },
    }

    qwen_client = QwenClient()
    raw_payload = qwen_client.call_function(messages, function_schema)

    if not raw_payload:
        return deepcopy(existing_profile) if existing_profile else {}

    try:
        extracted_model = UserProfile.model_validate(raw_payload)
    except ValidationError as exc:
        logger.warning("Discarding invalid profile update from QWEN backend: %s", exc)
        return deepcopy(existing_profile) if existing_profile else {}

    extracted_dict = extracted_model.model_dump(by_alias=True, exclude_none=True)

    merged_profile = deepcopy(existing_profile) if existing_profile else {}
    for key, value in extracted_dict.items():
        if key in {"Likes", "Dislikes", "Emotional Needs"}:
            merged_profile[key] = _merge_list_values(merged_profile.get(key, []), value)
        else:
            merged_profile[key] = value

    return merged_profile


def _extract_segments_qwen(
    chat_messages: List[Dict[str, str]],
    existing_segments: List[Dict[str, Any]],
    conversation_num: int,
    timestamp: str = ""
) -> List[Dict[str, Any]]:  # pragma: no cover - relies on external service
    existing_segments_json = json.dumps(existing_segments or [], ensure_ascii=False)
    system_prompt = (
        "Summarise the conversation chronologically into concise topical segments for downstream LLM retrieval. Emit a segment "
        "whenever the dialogue shifts to a new topic or subtask, and include both the user's statements and the assistant's suggestions "
        "relevant to that topic. Keep each summary under 30 tokens, capture emotional updates as standalone segments when they describe "
        "the user's state, and group all facts from the same topic together. Avoid mixing unrelated topics and do not duplicate segments "
        f"already present. Existing segments JSON: {existing_segments_json}."
    )

    messages = [{"role": "system", "content": system_prompt}] + chat_messages

    function_schema = {
        "name": "store_conversation_segments",
        "description": "Return topical conversation segment summaries for storage.",
        "parameters": {
            "type": "object",
            "properties": {
                "segments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string"}
                        },
                        "required": ["summary"],
                    },
                }
            },
            "required": ["segments"],
        },
    }

    qwen_client = QwenClient()
    raw_payload = qwen_client.call_function(messages, function_schema)

    payload_segments = raw_payload.get("segments") if isinstance(raw_payload, dict) else None
    if payload_segments is None:
        return deepcopy(existing_segments) if existing_segments else []

    try:
        extracted_segments = SegmentCollection(segments=payload_segments).segments
    except ValidationError as exc:
        logger.warning("Discarding invalid segments from QWEN backend: %s", exc)
        return deepcopy(existing_segments) if existing_segments else []

    merged_segments = deepcopy(existing_segments) if existing_segments else []
    existing_summaries = {
        (seg.get("summary", "") if isinstance(seg, dict) else "").strip()
        for seg in merged_segments
    }

    for seg in extracted_segments:
        summary = seg.summary.strip()
        if not summary or summary in existing_summaries:
            continue

        next_index = (
            sum(1 for existing in merged_segments if existing.get("convr_num") == conversation_num) + 1
        )
        segment_entry: Dict[str, Any] = {
            "summary": summary,
            "convr_num": conversation_num,
            "convr_index": next_index,
        }
        # Add timestamp only for the first segment of this conversation
        if next_index == 1 and timestamp:
            segment_entry["timestamp"] = timestamp
        merged_segments.append(segment_entry)
        existing_summaries.add(summary)

    return merged_segments


# =============================================================================
# System prompt helpers
# =============================================================================


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
        summary = seg.get("summary", "").strip()
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
        '"""\nMemory System Prompt 管理模块\n用于存储经过对话学到的用户信息和对话片段\n这个文件由 mem_bank.py 自动更新\n"""\n\n'
        f'MEMORY_SYSTEM_PROMPT = """{escaped}"""\n'
    )
    filepath.write_text(python_code, encoding="utf-8")


def _parse_memory_system_prompt(prompt_str: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    profile: Dict[str, Any] = {}
    segments: List[Dict[str, Any]] = []

    if not prompt_str:
        return profile, segments

    lines = [line.strip() for line in prompt_str.splitlines() if line.strip()]
    section = None

    for line in lines:
        if line.startswith("关于用户的信息"):
            section = "profile"
            continue
        if line.startswith("对话历史片段"):
            section = "segments"
            continue

        if section == "profile" and line.startswith("-"):
            # Example: "- 名字: Jacob"
            try:
                label, value = line[1:].split(":", 1)
                label = label.strip()
                value = value.strip()
            except ValueError:
                continue

            reverse_map = {v: k for k, v in PROFILE_KEY_MAP.items()}
            key = reverse_map.get(label)
            if not key:
                continue
            if key in {"Likes", "Dislikes", "Emotional Needs"}:
                profile[key] = [item.strip() for item in value.split("，") if item.strip()]
            else:
                profile[key] = value

        elif section == "segments" and line.startswith("CONVR"):
            try:
                tag, summary = line.split(".", 1)
                summary = summary.strip()
                
                # Handle format: CONVR1-1[timestamp] or CONVR1-1
                # Extract the tag part before the dot
                tag_clean = tag.strip()
                
                # Remove timestamp if present: CONVR1-1[2025-10-27 17:30:00] -> CONVR1-1
                if "[" in tag_clean:
                    tag_clean = tag_clean.split("[")[0]
                    # Extract timestamp for potential future use
                    timestamp_part = tag.split("[")[1].rstrip("]") if "[" in tag else ""
                else:
                    timestamp_part = ""
                
                # Parse CONVR{num}-{index}
                convr_part = tag_clean.replace("CONVR", "")
                parts = convr_part.split("-")
                if len(parts) != 2:
                    continue
                    
                convr_num = int(parts[0])
                convr_index = int(parts[1])
            except (ValueError, AttributeError, IndexError):
                continue

            segment_entry: Dict[str, Any] = {
                "summary": summary,
                "convr_num": convr_num,
                "convr_index": convr_index,
            }
            # Store timestamp if it exists (for first segment of each conversation)
            if timestamp_part:
                segment_entry["timestamp"] = timestamp_part
            
            segments.append(segment_entry)

    return profile, segments


def load_memory_from_system_prompt_file(path: Path | str = MEMORY_SYSTEM_PROMPT_FILE) -> Dict[str, Any]:
    filepath = Path(path)
    if not filepath.exists():
        return {"User Profile": {}, "Previous Conversation Segments": []}

    # Invalidate import caches to ensure we reload the file if it changed
    importlib.invalidate_caches()
    
    spec = importlib.util.spec_from_file_location("_memory_system_prompt_module", filepath)
    if spec is None or spec.loader is None:
        return {"User Profile": {}, "Previous Conversation Segments": []}

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]

    prompt_str = getattr(module, "MEMORY_SYSTEM_PROMPT", "")
    profile, segments = _parse_memory_system_prompt(prompt_str)
    return {
        "User Profile": profile or {},
        "Previous Conversation Segments": segments or [],
    }


# =============================================================================
# Incremental memory update helpers (currently unused but preserved)
# =============================================================================


def incrementally_update_memory(new_messages: list, existing_memory: Optional[dict] = None) -> dict:
    """Incrementally update memory using the configured backend.

    Note: Real-time invocation is currently disabled; this helper is preserved for
    future use when we re-enable periodic background processing.
    """

    existing_memory = existing_memory or {"User Profile": {}, "Previous Conversation Segments": []}

    chat_messages = _prepare_chat_messages(new_messages)
    if not chat_messages:
        return existing_memory

    backend = os.getenv("MEMORY_BACKEND", DEFAULT_BACKEND)
    conversation_num = peek_current_conversation_number() or 0

    try:
        if backend == "langmem":
            updated_profile = _extract_profile_langmem(chat_messages, existing_memory.get("User Profile", {}))
            updated_segments = _extract_segments_langmem(
                chat_messages,
                existing_memory.get("Previous Conversation Segments", []),
                conversation_num,
            )
        else:
            updated_profile = _extract_profile_qwen(chat_messages, existing_memory.get("User Profile", {}))
            updated_segments = _extract_segments_qwen(
                chat_messages,
                existing_memory.get("Previous Conversation Segments", []),
                conversation_num,
            )
    except AttributeError as exc:
        # Handle the specific case where typing.NotRequired is not available (Python < 3.11)
        if "NotRequired" in str(exc):
            logger.error("Incremental memory update failed due to incompatible Python version for langmem: %s. Falling back to QWEN.", exc)
            try:
                updated_profile = _extract_profile_qwen(chat_messages, existing_memory.get("User Profile", {}))
                updated_segments = _extract_segments_qwen(
                    chat_messages,
                    existing_memory.get("Previous Conversation Segments", []),
                    conversation_num,
                )
            except Exception as fallback_exc:
                logger.error("QWEN fallback also failed: %s", fallback_exc, exc_info=True)
                return existing_memory
        else:
            logger.error("Incremental memory update failed: %s", exc, exc_info=True)
            return existing_memory
    except Exception as exc:  # pragma: no cover - relies on external services
        logger.error("Incremental memory update failed: %s", exc, exc_info=True)
        return existing_memory

    return {
        "User Profile": updated_profile,
        "Previous Conversation Segments": updated_segments,
    }


# =============================================================================
# Main public API
# =============================================================================


def _load_conversation_messages(input_file: str | Path) -> List[Dict[str, Any]]:
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "messages" in data:
        return data["messages"]
    raise ValueError(f"Unexpected conversation history format in {input_file}")


def _write_transcript(output_file: str | Path, messages: List[Dict[str, Any]]) -> None:
    transcript = [
        {
            "role": message.get("role"),
            "content": message.get("content"),
            "timestamp": message.get("timestamp"),
        }
        for message in messages
        if message.get("role") in {"user", "assistant"}
    ]
    Path(output_file).write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")


def process_conversation_memory(
    input_file: str,
    output_transcript_file: str,
    output_system_prompt_file: str = str(MEMORY_SYSTEM_PROMPT_FILE),
    backend: Literal["langmem", "qwen"] = DEFAULT_BACKEND,
    conversation_num: Optional[int] = None,
) -> Dict[str, Any]:
    """Process the conversation history and update long-term memory files."""

    from datetime import datetime

    conversation_messages = _load_conversation_messages(input_file)
    chat_messages = _prepare_chat_messages(conversation_messages)
    if not chat_messages:
        raise ValueError("No user/assistant messages found for memory processing")

    Path(output_transcript_file).parent.mkdir(parents=True, exist_ok=True)
    _write_transcript(output_transcript_file, conversation_messages)

    existing_memory = load_memory_from_system_prompt_file(output_system_prompt_file)

    if conversation_num is None:
        conversation_num = _increment_counter()
    else:
        # Ensure counter is at least conversation_num to avoid duplicates
        current = peek_current_conversation_number()
        if conversation_num > current:
            CONVERSATION_COUNTER_FILE.write_text(str(conversation_num), encoding="utf-8")

    if backend not in {"langmem", "qwen"}:
        logger.warning("Unknown backend '%s', falling back to langmem", backend)
        backend = "langmem"

    profile_backend = _extract_profile_langmem
    segments_backend = _extract_segments_langmem

    if backend == "qwen":
        profile_backend = _extract_profile_qwen
        segments_backend = _extract_segments_qwen

    try:
        updated_profile = profile_backend(chat_messages, existing_memory.get("User Profile", {}))
    except AttributeError as exc:
        # Handle the specific case where typing.NotRequired is not available (Python < 3.11)
        if "NotRequired" in str(exc):
            logger.error("Primary backend profile extraction failed due to incompatible Python version for langmem: %s. Falling back to QWEN.", exc)
            updated_profile = _extract_profile_qwen(chat_messages, existing_memory.get("User Profile", {}))
        else:
            logger.error("Primary backend profile extraction failed (%s). Falling back to QWEN.", exc)
            updated_profile = _extract_profile_qwen(chat_messages, existing_memory.get("User Profile", {}))
    except Exception as exc:  # pragma: no cover - relies on external services
        logger.error("Primary backend profile extraction failed (%s). Falling back to QWEN.", exc)
        updated_profile = _extract_profile_qwen(chat_messages, existing_memory.get("User Profile", {}))

    # Capture timestamp when processing conversation memory
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        updated_segments = segments_backend(
            chat_messages,
            existing_memory.get("Previous Conversation Segments", []),
            conversation_num,
            timestamp,
        )
    except AttributeError as exc:
        # Handle the specific case where typing.NotRequired is not available (Python < 3.11)
        if "NotRequired" in str(exc):
            logger.error("Primary backend segment extraction failed due to incompatible Python version for langmem: %s. Falling back to QWEN.", exc)
            updated_segments = _extract_segments_qwen(
                chat_messages,
                existing_memory.get("Previous Conversation Segments", []),
                conversation_num,
                timestamp,
            )
        else:
            logger.error("Primary backend segment extraction failed (%s). Falling back to QWEN.", exc)
            updated_segments = _extract_segments_qwen(
                chat_messages,
                existing_memory.get("Previous Conversation Segments", []),
                conversation_num,
                timestamp,
            )
    except Exception as exc:  # pragma: no cover - relies on external services
        logger.error("Primary backend segment extraction failed (%s). Falling back to QWEN.", exc)
        updated_segments = _extract_segments_qwen(
            chat_messages,
            existing_memory.get("Previous Conversation Segments", []),
            conversation_num,
            timestamp,
        )

    memory_payload = {
        "User Profile": updated_profile,
        "Previous Conversation Segments": updated_segments,
        "conversation_number": conversation_num,
    }

    system_prompt_str = _build_system_prompt_string(memory_payload)
    _write_system_prompt_file(Path(output_system_prompt_file), system_prompt_str)

    return memory_payload


__all__ = [
    "process_conversation_memory",
    "incrementally_update_memory",
    "load_memory_from_system_prompt_file",
    "peek_current_conversation_number",
    "reset_conversation_counter",
]
