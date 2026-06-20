"""This module defines the nodes used in the Graph of Chatbot."""

from app.ai_layer.model import initialize_model, initialize_tool_enabled_model
from app.ai_layer.graph_states import GraphState
from app.ai_layer.prompts import get_system_message
from app.ai_layer.tools import tools_list

model = initialize_model()
model_with_tools = initialize_tool_enabled_model(model, tools_list)

sys_msg = get_system_message()


def chat_with_model(state: GraphState) -> str:
    """Node function that takes the messages from state and returns a response from the model."""
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def reasoner(state: GraphState):
    """Node function that takes the messages from state and returns a response from the model with tools enabled."""
    return {"messages": [model_with_tools.invoke([sys_msg] + state["messages"])]}
