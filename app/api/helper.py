"""Helper functions for the API endpoints."""

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AIMessageChunk
from app.ai_layer.agent import chatbot


def to_openai_format(messages):
    """Convert LangChain message format to OpenAI format."""
    formatted = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted.append({"role": "user", "content": msg.content})

        elif isinstance(msg, AIMessage):
            formatted.append({"role": "assistant", "content": msg.content})

        elif isinstance(msg, SystemMessage):
            formatted.append({"role": "system", "content": msg.content})

        # ToolMessage → ignored

    return formatted


async def chat_with_agent(message: str, session_id: str):
    """Chat with the agent and return the response in OpenAI format. and follow the async pattern."""
    results = await chatbot.ainvoke(
        {"messages": [{"role": "user", "content": message}]},
        {"configurable": {"thread_id": session_id}},
    )
    return to_openai_format(results["messages"])


async def chat_with_agent_stream(message: str, session_id: str):
    """Stream chat responses from the agent in OpenAI format."""
    try:
        async for msg, _ in chatbot.astream(
            {"messages": [{"role": "user", "content": message}]},
            {"configurable": {"thread_id": session_id}},
            stream_mode="messages",
        ):
            if isinstance(msg, AIMessageChunk) and msg.content:
                # escape newlines so they don't break SSE format
                content = msg.content.replace("\n", "\\n")
                yield f"data: {content}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:  # pylint: disable=broad-exception-caught
        yield f"data: ERROR: {str(e)}\n\n"
