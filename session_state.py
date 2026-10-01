"""Session management helpers for updating model instructions with memory."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional


class SessionState:
    """Coordinate session updates so new memory reaches the model promptly."""

    def __init__(
        self,
        conversation_history: Any,
        prompt_generator: Callable[[Any], str],
    ) -> None:
        self._conversation_history = conversation_history
        self._prompt_generator = prompt_generator
        self._conversation: Optional[Any] = None
        self._session_kwargs: Optional[Dict[str, Any]] = None
        self._last_refresh_time: float = 0.0
        self._last_consumed_memory_time: float = 0.0

    @property
    def last_refresh_time(self) -> float:
        return self._last_refresh_time

    def bind_conversation(
        self,
        conversation: Any,
        session_kwargs: Dict[str, Any],
        debug_printer: Callable[[str], None] = print,
    ) -> None:
        """Register the live conversation and log the initial payload."""
        self._conversation = conversation
        self._session_kwargs = dict(session_kwargs)
        self._last_refresh_time = time.time()

        instructions = self._session_kwargs.get("instructions", "")
        debug_printer(self._format_debug_message("initial", instructions))

    def refresh_if_needed(
        self,
        reason: str = "auto",
        debug_printer: Callable[[str], None] = print,
    ) -> bool:
        """Refresh session instructions if a memory update is pending."""
        if not self._conversation:
            return False

        history = self._conversation_history
        pending_refresh = getattr(history, "pending_prompt_refresh", False)
        last_memory_time = getattr(history, "last_memory_update", 0.0)
        if not pending_refresh and last_memory_time <= self._last_consumed_memory_time:
            return False

        new_prompt = self._prompt_generator(history)
        history.refresh_system_message(new_prompt)

        payload = dict(self._session_kwargs or {})
        payload["instructions"] = new_prompt
        self._conversation.update_session(**payload)

        self._session_kwargs = payload
        self._last_refresh_time = time.time()
        self._last_consumed_memory_time = last_memory_time
        if pending_refresh:
            history.pending_prompt_refresh = False

        debug_printer(self._format_debug_message(reason, new_prompt))
        return True

    def get_session_kwargs(self) -> Dict[str, Any]:
        """Return a copy of the most recent session kwargs."""
        return dict(self._session_kwargs or {})

    @staticmethod
    def _format_debug_message(reason: str, instructions: str) -> str:
        preview = instructions[:200].replace("\n", " ")
        return f"[DEBUG] Updated model instructions ({reason}) -> {preview}..."
