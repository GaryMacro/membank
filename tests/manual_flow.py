"""Manual simulation of assistant conversation flow for verification."""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from conversation_history import (
    conversation_history,
    initialize_history_with_memory,
    save_and_process_conversation,
)
from session_state import SessionState
from system_prompt import update_system_prompt_with_current_memory
from openai import OpenAI
from mem_bank import MEMORY_SYSTEM_PROMPT_FILE
from config import EXIT_WORDS


_API_KEY = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
if not _API_KEY:
    raise RuntimeError("QWEN_API_KEY, DASHSCOPE_API_KEY, or OPENAI_API_KEY must be set for manual flow.")

_BASE_URL = os.getenv("QWEN_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
_MODEL_ID = os.getenv("QWEN_MODEL", "qwen-plus")

_qwen_client = OpenAI(api_key=_API_KEY.strip(), base_url=_BASE_URL.strip())

EXIT_PHRASE = EXIT_WORDS[0] if EXIT_WORDS else "goodbye"


class DummyConversation:
    def __init__(self):
        self.updates = []

    def update_session(self, **kwargs):
        self.updates.append(kwargs)
        print(f"[DummyConversation] update_session called with instructions snippet: {kwargs.get('instructions', '')[:60]}")


def _build_model_messages(session_state: SessionState) -> List[Dict[str, str]]:
    instructions = session_state.get_session_kwargs().get("instructions", "")
    payload: List[Dict[str, str]] = []
    if instructions:
        payload.append({"role": "system", "content": instructions})

    for message in conversation_history.messages:
        role = message.get("role")
        content = str(message.get("content") or "")
        if not content:
            continue
        if role == "system":
            # Avoid duplicating the system prompt, already injected above
            continue
        if role == "summariser output":
            payload.append({"role": "system", "content": content})
        else:
            payload.append({"role": str(role), "content": content})

    return payload


def _invoke_qwen(session_state: SessionState, tag: str) -> str:
    payload = _build_model_messages(session_state)
    print(f"\n[MODEL INPUT:{tag}]")
    for item in payload:
        preview = item["content"] if len(item["content"]) < 400 else item["content"][:400] + "..."
        print(f"{item['role'].upper()}: {preview}")

    api_payload = cast(List[Dict[str, str]], payload)
    response = _qwen_client.chat.completions.create(
        model=_MODEL_ID,
        messages=api_payload,  # type: ignore[arg-type]
        temperature=float(os.getenv("QWEN_TEMPERATURE", "0.3")),
    )
    assistant_message = response.choices[0].message.content or ""
    print(f"\n[MODEL OUTPUT:{tag}] {assistant_message}\n")
    return assistant_message.strip()


def _run_user_turn(session_state: SessionState, user_text: str, tag: str) -> None:
    conversation_history.add_message("user", user_text)
    assistant_text = _invoke_qwen(session_state, tag)
    conversation_history.add_message("assistant", assistant_text)


def _load_history_messages(history_path: str) -> List[Dict[str, Any]]:
    with Path(history_path).open("r", encoding="utf-8") as history_file:
        payload = json.load(history_file)
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected history payload type: {type(payload)!r}")
    return payload


def _ensure_summary_in_history(history_path: str) -> str:
    messages = _load_history_messages(history_path)
    summary_entries = [msg for msg in messages if msg.get("role") == "summariser output"]
    if not summary_entries:
        raise AssertionError(f"No summariser output present in {history_path}")
    summary_text = str(summary_entries[-1].get("content", "")).strip()
    if not summary_text:
        raise AssertionError(f"Summariser output empty in {history_path}")
    return summary_text


def _exit_and_summarize(
    session_state: SessionState,
    debug_messages: List[str],
    reason: str,
) -> tuple[str, str]:
    conversation_history.add_message("user", EXIT_PHRASE)
    save_and_process_conversation()
    while getattr(conversation_history, "processing_memory", False):
        time.sleep(0.2)

    summary = conversation_history.summarize_recent_conversation(
        max_tokens_before_summary=32,
        max_summary_tokens=64,
    )
    if not summary:
        raise RuntimeError(f"Expected non-empty summary during {reason}")

    history_path = conversation_history.save_history()
    if not history_path:
        raise RuntimeError("Failed to persist conversation history after exit")

    summary_from_disk = _ensure_summary_in_history(history_path)
    if summary_from_disk != summary:
        print("[WARN] File summary differs from in-memory summary; using file version for checks.")
        summary = summary_from_disk

    conversation_history.pending_prompt_refresh = True
    conversation_history.last_memory_update = time.time()
    session_state.refresh_if_needed(f"{reason}-summary", debug_messages.append)
    if debug_messages:
        print(debug_messages[-1])

    return summary, history_path


def run_simulation():
    prompt_path = Path(MEMORY_SYSTEM_PROMPT_FILE)
    original_prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else None

    conversation_history.clear_history()
    initialize_history_with_memory()

    debug_messages: list[str] = []
    initial_prompt = update_system_prompt_with_current_memory(conversation_history)
    session_state = SessionState(conversation_history, update_system_prompt_with_current_memory)
    session_state.bind_conversation(
        DummyConversation(),
        {"instructions": initial_prompt, "voice": "TestVoice"},
        debug_messages.append,
    )
    print(debug_messages[-1])

    try:
        print("\n=== Conversation 1: Wake and share family detail ===")
        _run_user_turn(session_state, "hello lingling", "conv1-turn-1")
        _run_user_turn(session_state, "My aunt's name is Sarah.", "conv1-turn-2")
        _run_user_turn(
            session_state,
            "She loves baking lemon tarts and always shares extra slices with me.",
            "conv1-turn-3",
        )
        summary1, history_path1 = _exit_and_summarize(session_state, debug_messages, "conv1")
        print(f"\n[Summary after conversation 1]\n{summary1}")
        print(f"History written to {history_path1}")

        if "Sarah" in summary1:
            print("Verified: Sarah detail captured in summary 1.")

        print("\n=== Conversation 2: Wake and add uncle detail ===")
        _run_user_turn(session_state, "Hi Lingling, it's Mark again.", "conv2-turn-1")
        _run_user_turn(session_state, "My uncle's name is Dan.", "conv2-turn-2")
        _run_user_turn(
            session_state,
            "He collects rare stamps with me on Sunday afternoons.",
            "conv2-turn-3",
        )
        summary2, history_path2 = _exit_and_summarize(session_state, debug_messages, "conv2")
        print(f"\n[Summary after conversation 2]\n{summary2}")
        print(f"History updated at {history_path2}")

        if "Sarah" not in summary2 or "Dan" not in summary2:
            raise AssertionError("Summary after conversation 2 must retain Sarah and include Dan.")
        print("Summary extension verified: Conversation 2 summary includes Sarah and Dan.")

        print("\n=== Conversation 3: Wake and confirm memory recall ===")
        _run_user_turn(session_state, "Hi again, thanks for remembering my family.", "conv3-turn-1")
        _run_user_turn(
            session_state,
            "My cousin Lily is joining us for the next tart-baking evening.",
            "conv3-turn-2",
        )
        _run_user_turn(
            session_state,
            "Could you remind me what dessert Aunt Sarah loves making?",
            "conv3-turn-3",
        )
        summary3, history_path3 = _exit_and_summarize(session_state, debug_messages, "conv3")
        print(f"\n[Summary after conversation 3]\n{summary3}")
        print(f"History refreshed at {history_path3}")

        if "Lily" in summary3:
            print("Summary after conversation 3 reflects new Lily detail.")

        print("\nVerification complete: three wake/exit cycles processed successfully.")

    finally:
        if original_prompt is None:
            if prompt_path.exists():
                prompt_path.unlink()
        else:
            prompt_path.write_text(original_prompt, encoding="utf-8")


if __name__ == "__main__":
    run_simulation()
