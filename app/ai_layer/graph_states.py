"""This module defines the graph states used in the AI layer of the application."""

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated


class GraphState(TypedDict):
    """Defines the state of the graph, which includes a list of messages."""

    messages: Annotated[list[BaseMessage], add_messages]
