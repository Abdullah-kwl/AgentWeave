"""This module defines the main agent that will be used to interact
with the language model using a LangGraph based approach."""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from app.ai_layer.graph_states import GraphState
from app.ai_layer.nodes import chat_with_model

checkpoint_saver = InMemorySaver()

graph = StateGraph(GraphState)
graph.add_node("chat", chat_with_model)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

chatbot = graph.compile(checkpointer=checkpoint_saver)
