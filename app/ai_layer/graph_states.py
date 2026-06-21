"""This module defines the graph states used in the AI layer of the application."""

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, NotRequired, TypedDict


class GraphState(TypedDict):
    """Defines the state of the graph.

    The STM fields are ``NotRequired`` so the graph can be invoked for a new
    session without pre-populating them; each STM helper falls back to safe
    defaults (``""`` / ``0``) when they are absent from the checkpoint.

    Attributes:
        messages: Full accumulated message list managed by LangGraph's
            ``add_messages`` reducer.
        summary: Rolling plain-text summary of messages that have left the
            live context window.  Empty string when no summarisation has
            occurred yet.
        summarize_at: Total message count at which the next summarisation
            pass should be triggered.  Defaults to ``THRESHOLD`` on first use.
        context_start: Index into ``messages`` marking the start of the live
            context window.  Messages before this index have been summarised.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    summary: NotRequired[str]
    summarize_at: NotRequired[int]
    context_start: NotRequired[int]
