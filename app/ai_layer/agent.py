"""This module defines the main agent that will be used to interact
with the language model using a LangGraph based approach."""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

# from langchain_core.messages import HumanMessage
from app.ai_layer.graph_states import GraphState
from app.ai_layer.nodes import chat_with_model

checkpoint_saver = InMemorySaver()

graph = StateGraph(GraphState)
graph.add_node("chat", chat_with_model)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

chatbot = graph.compile(checkpointer=checkpoint_saver)

# if __name__ == "__main__":
#     # print(chatbot.get_graph().draw_ascii())
#     config={"configurable": {"thread_id": 1}}
#     initial_state = {"messages": [HumanMessage(content="my name is ali i like icecreame specially chocolate flavor")]}
#     result = chatbot.invoke(initial_state, config=config)
#     # print(result)
#     print(result["messages"][-1].content)
#     print("----"*10)

#     initial_state = {"messages": [HumanMessage(content="what you know about me")]}
#     result = chatbot.invoke(initial_state, config=config)
#     # print(result)
#     print(result["messages"][-1].content)
