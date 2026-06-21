"""Short-term memory (STM) helpers for managing conversation context.

The STM layer prevents unbounded token growth by maintaining a rolling
summary of older messages while keeping only a recent window in the live
context sent to the model.  When the total message count reaches
``THRESHOLD``, the slice of messages that is about to leave the window is
summarised by the plain (no-tools) model and stored in the graph state.
Subsequent calls prepend that summary as a second ``SystemMessage`` so the
model retains past context without re-sending every historical token.
"""

import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.ai_layer.graph_states import GraphState
from app.utils.config import stm_config

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _role(msg: BaseMessage) -> str:
    """Return a human-readable role label for a LangChain message object."""
    if isinstance(msg, HumanMessage):
        return "User"
    if isinstance(msg, AIMessage):
        return "Assistant"
    if isinstance(msg, SystemMessage):
        return "System"
    return type(msg).__name__


# ── Public helpers ────────────────────────────────────────────────────────────


def get_llm_context(state: GraphState) -> list[BaseMessage]:
    """Build the message list that will actually be sent to the LLM.

    When a rolling summary exists it is prepended as a ``SystemMessage`` so
    the model understands past context without receiving every old token.
    Only the recent slice (from ``context_start`` onward) of the full message
    list is appended after the summary.

    The resulting list is intended to be concatenated after the primary system
    message in the caller, yielding the two-system-message pattern::

        [primary_sys_msg, summary_sys_msg (optional), *recent_messages]

    Args:
        state: Current graph state.  The optional STM fields ``summary`` and
            ``context_start`` default to ``""`` and ``0`` respectively when
            absent (e.g. on the first turn of a new session).

    Returns:
        A list of ``BaseMessage`` objects ready to be sent to the model.
    """
    context: list[BaseMessage] = []

    summary: str = state.get("summary", "")  # type: ignore[call-overload]
    context_start: int = state.get("context_start", 0)  # type: ignore[call-overload]

    if summary:
        context.append(SystemMessage(content=f"Conversation summary so far:\n{summary}"))

    context.extend(state["messages"][context_start:])
    return context


def summarize_and_trim(state: GraphState, model: BaseChatModel) -> dict:
    """Compress the messages that are leaving the context window.

    Reads the slice of messages between the current ``context_start`` and the
    new window boundary (``total - KEEP_RECENT``), asks *model* to produce an
    updated rolling summary, then returns a partial state dict containing the
    three STM fields so the caller can merge them back into the graph state.

    When there are no messages to summarise (e.g. the window boundary lands
    before ``context_start``) only ``summarize_at`` is updated.

    Args:
        state: Current graph state.
        model: Plain (no tools) language model used to generate the summary.
            Kept separate from the tool-enabled model to avoid tool-call
            noise in the summarisation prompt.

    Returns:
        Partial state dict with one or more of: ``summary``, ``summarize_at``,
        ``context_start``.
    """
    total: int = len(state["messages"])
    context_start: int = state.get("context_start", 0)  # type: ignore[call-overload]
    existing_summary: str = state.get("summary", "")  # type: ignore[call-overload]

    new_ctx_start: int = total - stm_config.keep_recent
    to_summarize: list[BaseMessage] = state["messages"][context_start:new_ctx_start]

    if not to_summarize:
        return {"summarize_at": total + stm_config.summarize_step}

    history_text: str = "\n".join(f"{_role(m)}: {m.content}" for m in to_summarize)

    # fmt: off
    if existing_summary:
        prompt = (
            f"You are maintaining a rolling summary of a conversation.\n\n"
            f"Existing summary:\n{existing_summary}\n\n"
            f"New messages to incorporate:\n{history_text}\n\n"
            f"Write an updated, concise summary that captures all important "
            f"context from both the existing summary and the new messages. "
            f"Plain text only, no bullet points."
        )
    else:
        prompt = (
            f"Summarize the following conversation excerpt concisely, "
            f"capturing all important context. Plain text only.\n\n{history_text}"
        )
    # fmt: on

    # callbacks=[] detaches this internal call from LangGraph's streaming
    # machinery so the summarisation output is never forwarded to the client.
    response = model.invoke([HumanMessage(content=prompt)], config={"callbacks": []})
    # Strip internal chain-of-thought blocks emitted by reasoning models.
    new_summary: str = _THINK_RE.sub("", response.content).strip()

    return {
        "summary": new_summary,
        "summarize_at": total + stm_config.summarize_step,
        "context_start": new_ctx_start,
    }
