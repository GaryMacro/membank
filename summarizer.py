"""Conversation summariser using langmem's short-term helper."""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
from langmem.short_term import RunningSummary, summarize_messages
from pydantic import SecretStr


SUMMARY_INSTRUCTIONS = (
    "You compress user-assistant dialogue into persistent memory notes. "
    "Use as few tokens as possible while staying factual. "
    "Only use bullet points when there are multiple distinct facts; otherwise respond with a single concise sentence. "
    "Capture user-specific facts, commitments, and open questions only. No emojis, no quotes, no stylistic filler."
)

SUMMARY_INITIAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(SUMMARY_INSTRUCTIONS),
        MessagesPlaceholder("messages", optional=True),
        HumanMessagePromptTemplate.from_template("Produce the condensed memory now."),
    ]
)

SUMMARY_EXISTING_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(SUMMARY_INSTRUCTIONS),
        MessagesPlaceholder("messages", optional=True),
        HumanMessagePromptTemplate.from_template(
            "Update the memory with any new salient facts following the same rules."
        ),
    ]
)


def _merge_summary_text(existing_summary: str, new_summary: str) -> str:
    existing_summary = (existing_summary or "").strip()
    new_summary = (new_summary or "").strip()

    if not existing_summary:
        return new_summary
    if not new_summary:
        return existing_summary

    if existing_summary.lower() in new_summary.lower():
        return new_summary
    if new_summary.lower() in existing_summary.lower():
        return existing_summary

    def _extract_facts(summary_text: str) -> list[str]:
        lines = [line.strip() for line in summary_text.splitlines() if line.strip()]
        if not lines:
            return []
        if len(lines) == 1 and not lines[0].startswith("- "):
            return [lines[0]]
        facts: list[str] = []
        for line in lines:
            fact = line[2:].strip() if line.startswith("- ") else line
            if fact:
                facts.append(fact)
        return facts or [summary_text.strip()]

    combined: list[str] = []
    seen: set[str] = set()
    for segment in (existing_summary, new_summary):
        for fact in _extract_facts(segment):
            key = fact.lower()
            if key in seen:
                continue
            seen.add(key)
            combined.append(fact)

    if not combined:
        return new_summary or existing_summary
    if len(combined) == 1:
        return combined[0]
    return "\n".join(f"- {fact}" for fact in combined)


def _make_model() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv("SUMMARY_MODEL", "gpt-4o")

    if not api_key:
        api_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        base_url = os.getenv("QWEN_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL") or base_url
        model_name = os.getenv("QWEN_SUMMARY_MODEL", os.getenv("QWEN_MODEL", "qwen-plus"))

    if not api_key:
        raise RuntimeError("No API key found for summariser model")

    return ChatOpenAI(  # type: ignore[arg-type]
        api_key=SecretStr(api_key.strip()),
        base_url=base_url.strip() if base_url else None,
        model=model_name.strip(),
        temperature=0.0,
        streaming=False,
    )


try:
    model: Optional[ChatOpenAI] = _make_model()
    summarization_model = model.bind(max_tokens=128) if model else None
    _init_error: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - environment dependent
    model = None
    summarization_model = None
    _init_error = exc


def generate_summary(
    messages: Iterable[dict[str, str]],
    *,
    max_tokens_before_summary: int | None = None,
    max_tokens: int = 256,
    max_summary_tokens: int = 128,
    running_summary: RunningSummary | None = None,
) -> tuple[str, RunningSummary | None]:
    """Return a short summary string for the provided chat messages."""
    raw_messages = [dict(message) for message in messages]
    if not raw_messages:
        return "", running_summary

    if not model or not summarization_model:
        return _fallback_summary(raw_messages), running_summary

    structured: list[BaseMessage] = []
    for idx, message in enumerate(raw_messages):
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = message.get("content", "")
        msg_id = message.get("id") or message.get("timestamp") or f"msg-{idx}"
        if role == "user":
            structured.append(HumanMessage(content=content, id=msg_id))
        else:
            structured.append(AIMessage(content=content, id=msg_id))

    if not structured:
        return "", running_summary

    try:
        payload = cast(list[Any], structured)
        summary_result = summarize_messages(
            payload,
            running_summary=running_summary,
            model=summarization_model,
            max_tokens=max_tokens,
            max_tokens_before_summary=max_tokens_before_summary,
            max_summary_tokens=max_summary_tokens,
            initial_summary_prompt=SUMMARY_INITIAL_PROMPT,
            existing_summary_prompt=SUMMARY_EXISTING_PROMPT,
        )
        updated_running_summary = summary_result.running_summary or running_summary
        summary_text = ""
        if updated_running_summary:
            summary_text = (updated_running_summary.summary or "").strip()
            if running_summary and running_summary.summary and summary_text:
                merged_summary = _merge_summary_text(running_summary.summary, summary_text)
                if merged_summary != summary_text:
                    summary_text = merged_summary
                    updated_running_summary = RunningSummary(
                        summary=summary_text,
                        summarized_message_ids=set(updated_running_summary.summarized_message_ids),
                        last_summarized_message_id=updated_running_summary.last_summarized_message_id,
                    )
        if summary_text:
            return summary_text, updated_running_summary
        return _direct_summary(raw_messages, structured, updated_running_summary)
    except Exception as exc:  # pragma: no cover - network call fallback
        return _fallback_summary(raw_messages, str(exc)), running_summary


def _direct_summary(
    raw_messages: list[dict[str, str]],
    structured: list[BaseMessage],
    running_summary: RunningSummary | None,
) -> tuple[str, RunningSummary | None]:
    if not model or not summarization_model:
        return _fallback_summary(raw_messages), running_summary

    try:
        if running_summary and running_summary.summary:
            formatted = SUMMARY_EXISTING_PROMPT.format_messages(messages=structured)
        else:
            formatted = SUMMARY_INITIAL_PROMPT.format_messages(messages=structured)

        response = (summarization_model or model).invoke(formatted)
        content = getattr(response, "content", "")
        if isinstance(content, list):
            content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
        summary_text = str(content).strip()
        if not summary_text:
            return _fallback_summary(raw_messages), running_summary

        if running_summary and running_summary.summary:
            summary_text = _merge_summary_text(running_summary.summary, summary_text)

        summarized_ids: set[str] = set()
        last_id: str | None = None
        for message in structured:
            msg_id = getattr(message, "id", None)
            if isinstance(msg_id, str):
                summarized_ids.add(msg_id)
                last_id = msg_id

        combined_ids = set(running_summary.summarized_message_ids) if running_summary else set()
        combined_ids |= summarized_ids
        if not last_id and running_summary:
            last_id = running_summary.last_summarized_message_id

        new_running_summary = RunningSummary(
            summary=summary_text,
            summarized_message_ids=combined_ids,
            last_summarized_message_id=last_id,
        )
        return summary_text, new_running_summary
    except Exception as exc:  # pragma: no cover - network call fallback
        return _fallback_summary(raw_messages, str(exc)), running_summary


def _fallback_summary(messages: list[dict[str, str]], reason: str | None = None) -> str:
    prefix = f"Summary unavailable ({reason})" if reason else "Summary unavailable"
    lines = []
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "").strip()
        if role in {"user", "assistant"} and content:
            lines.append(f"{role}: {content}")
    if not lines:
        return ""
    return prefix + "\n" + "\n".join(lines)
