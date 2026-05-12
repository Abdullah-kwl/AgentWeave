"""This module defines the nodes used in the Graph of Chatbot."""

from app.ai_layer.model import model
from app.ai_layer.graph_states import GraphState


def chat_with_model(state: GraphState) -> str:
    """Node function that takes the messages from state and returns a response from the model."""
    response = model.invoke(state["messages"])
    return {"messages": [response]}
