"""This module defines the main agent that will be used to interact
with the language model using a LangGraph based approach."""

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from app.ai_layer.graph_states import GraphState
from app.ai_layer.nodes import chat_with_model

graph = StateGraph(GraphState)
graph.add_node("chat", chat_with_model)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

chatbot = graph.compile()

if __name__ == "__main__":
    # print(chatbot.get_graph().draw_ascii())
    initial_state = {"messages": [HumanMessage(content="what is the meaning of life?")]}
    result = chatbot.invoke(initial_state)
    print(result)
