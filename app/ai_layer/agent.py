"""This module defines the main agent that will be used to interact
with the language model using a LangGraph based approach."""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.ai_layer.graph_states import GraphState
from app.ai_layer.nodes import reasoner
from app.ai_layer.tools import tools_list

# Graph
builder = StateGraph(GraphState)
checkpoint_saver = InMemorySaver()

# Add nodes
builder.add_node("reasoner", reasoner)
builder.add_node("tools", ToolNode(tools_list))  # for the tools

# Add edges
builder.add_edge(START, "reasoner")
builder.add_conditional_edges(
    "reasoner",
    tools_condition,
)
builder.add_edge("tools", "reasoner")

chatbot = builder.compile(checkpointer=checkpoint_saver)
