"""Helper functions for the API endpoints."""

import re

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage

from app.ai_layer.agent import chatbot

# Matches a complete <think>…</think> block including newlines.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think_tags(text: str) -> str:
    """Remove <think>…</think> reasoning blocks from model output.

    Reasoning models (e.g. qwen/qwen3.6-27b on Groq) emit their internal
    chain-of-thought wrapped in these tags.  They must be stripped before
    the text is returned to a client or stored as a summary.

    Args:
        text: Raw model output that may contain one or more think blocks.

    Returns:
        Cleaned text with all think blocks removed and surrounding
        whitespace stripped.
    """
    return _THINK_RE.sub("", text).strip()


def to_openai_format(messages):
    """Convert LangChain message format to OpenAI format.

    Think-tag blocks are stripped from assistant messages so that internal
    model reasoning never leaks into the API response.

    Args:
        messages: List of LangChain ``BaseMessage`` objects.

    Returns:
        List of dicts with ``role`` and ``content`` keys.
    """
    formatted = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted.append({"role": "user", "content": msg.content})

        elif isinstance(msg, AIMessage):
            formatted.append({"role": "assistant", "content": _strip_think_tags(msg.content)})

        elif isinstance(msg, SystemMessage):
            formatted.append({"role": "system", "content": msg.content})

        # ToolMessage → ignored

    return formatted


async def chat_with_agent(message: str, session_id: str):
    """Chat with the agent and return the full response in OpenAI format.

    Args:
        message: The user's input text.
        session_id: Identifies the conversation thread for the checkpointer.

    Returns:
        List of message dicts in OpenAI format (think blocks stripped).
    """
    results = await chatbot.ainvoke(
        {"messages": [{"role": "user", "content": message}]},
        {"configurable": {"thread_id": session_id}},
    )
    return to_openai_format(results["messages"])


async def chat_with_agent_stream(message: str, session_id: str):
    """Stream chat responses from the agent as Server-Sent Events.

    Reasoning models emit ``<think>…</think>`` blocks at the start of their
    output.  This generator tracks whether the current chunk falls inside a
    think block and discards those tokens, forwarding only the visible reply.

    Args:
        message: The user's input text.
        session_id: Identifies the conversation thread for the checkpointer.

    Yields:
        SSE-formatted strings (``data: <content>\\n\\n``).
        Newlines within a chunk are escaped as ``\\n``.
        The stream ends with ``data: [DONE]\\n\\n``.
    """
    try:
        llm_config = {"configurable": {"thread_id": session_id}}
        msg_state = {"messages": [HumanMessage(content=message)]}

        # State machine: track whether we are currently inside a <think> block.
        in_think_block: bool = False
        think_buffer: str = ""

        async for chunk in chatbot.astream(
            msg_state,
            config=llm_config,
            version="v2",
            stream_mode=["messages"],
        ):
            if chunk["type"] != "messages":
                continue

            msg, _ = chunk["data"]
            if not (isinstance(msg, AIMessageChunk) and msg.content):
                continue

            raw: str = msg.content

            if in_think_block:
                # Accumulate until we see the closing tag.
                think_buffer += raw
                if "</think>" in think_buffer:
                    in_think_block = False
                    after = think_buffer.split("</think>", 1)[1]
                    think_buffer = ""
                    if after:
                        yield f"data: {after.replace(chr(10), chr(92) + 'n')}\n\n"
            else:
                if "<think>" in raw:
                    # Emit anything before the opening tag, then enter think mode.
                    before, rest = raw.split("<think>", 1)
                    if before:
                        yield f"data: {before.replace(chr(10), chr(92) + 'n')}\n\n"
                    if "</think>" in rest:
                        # Entire think block arrived in one chunk — skip it.
                        after = rest.split("</think>", 1)[1]
                        if after:
                            yield f"data: {after.replace(chr(10), chr(92) + 'n')}\n\n"
                    else:
                        in_think_block = True
                        think_buffer = rest
                else:
                    yield f"data: {raw.replace(chr(10), chr(92) + 'n')}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:  # pylint: disable=broad-exception-caught
        yield f"data: ERROR: {str(e)}\n\n"


async def old_chat_history(session_id: str):
    """Retrieve the chat history for a given session.

    Args:
        session_id: Identifies the conversation thread for the checkpointer.

    Returns:
        List of message dicts in OpenAI format (think blocks stripped).
    """
    llm_config = {"configurable": {"thread_id": session_id}}
    old_messages = chatbot.get_state(config=llm_config)
    return to_openai_format(old_messages.values["messages"])
