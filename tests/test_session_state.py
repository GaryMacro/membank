import json
import time
from pathlib import Path

import pytest

from conversation_history import ConversationHistoryManager
from session_state import SessionState


class DummyConversation:
    def __init__(self):
        self.updates = []

    def update_session(self, **kwargs):
        self.updates.append(kwargs)


class DummyHistory:
    def __init__(self):
        self.last_memory_update = 0.0
        self.pending_prompt_refresh = False
        self._prompt = ""
        self.fact = ""

    def refresh_system_message(self, prompt_text: str) -> None:
        self._prompt = prompt_text


@pytest.fixture
def dummy_history():
    history = DummyHistory()
    history.fact = "Aunt Sarah"
    return history


def test_assistant_messages_are_committed(tmp_path):
    manager = ConversationHistoryManager(history_dir=str(tmp_path))
    manager.add_message("system", "initial")

    manager.capture_assistant_delta("Hello")
    manager.capture_assistant_delta(" world")
    committed = manager.commit_assistant_message()

    assert committed == "Hello world"

    path = manager.save_history()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assistant_messages = [msg for msg in data if msg["role"] == "assistant"]
    assert len(assistant_messages) == 1
    assert assistant_messages[0]["content"] == "Hello world"


def test_session_state_refreshes_when_memory_updates(dummy_history):
    def prompt_generator(_: DummyHistory) -> str:
        return f"Remember: {dummy_history.fact}"

    session = SessionState(dummy_history, prompt_generator)
    convo = DummyConversation()
    debug_messages = []

    session.bind_conversation(
        convo,
        {"instructions": "Initial prompt", "voice": "TestVoice"},
        debug_messages.append,
    )

    dummy_history.pending_prompt_refresh = True
    dummy_history.last_memory_update = time.time() + 1

    refreshed = session.refresh_if_needed("memory", debug_messages.append)

    assert refreshed is True
    assert debug_messages
    assert "memory" in debug_messages[-1]
    assert "Aunt Sarah" in convo.updates[-1]["instructions"]
    assert dummy_history.pending_prompt_refresh is False

    prev_updates = len(convo.updates)
    refreshed_again = session.refresh_if_needed("no-change", debug_messages.append)
    assert refreshed_again is False
    assert len(convo.updates) == prev_updates
