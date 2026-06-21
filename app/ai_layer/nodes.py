"""This module defines the nodes used in the Graph of Chatbot."""

from app.ai_layer.graph_states import GraphState
from app.ai_layer.model import initialize_model, initialize_tool_enabled_model
from app.ai_layer.prompts import get_system_message
from app.ai_layer.stm import get_llm_context, summarize_and_trim
from app.utils.config import stm_config
from app.ai_layer.tools import tools_list

model = initialize_model()
model_with_tools = initialize_tool_enabled_model(model, tools_list)

_sys_msg = get_system_message()


def chat_with_model(state: GraphState) -> dict:
    """Invoke the plain model (no tools) with the full message history.

    Args:
        state: Current graph state.

    Returns:
        Partial state dict containing the new assistant message.
    """
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def reasoner(state: GraphState) -> dict:
    """Invoke the tool-enabled model with STM-managed context.

    Before calling the LLM the node checks whether the total message count has
    reached the summarisation threshold.  If so it compresses the oldest
    messages into a rolling summary and advances the context window pointer.
    Either way it builds the effective context via :func:`get_llm_context` and
    prefixes it with the primary system message.

    Two ``SystemMessage`` objects can appear in the final prompt sent to the
    LLM:

    1. ``sys_msg`` — the fixed assistant-role instruction, always first.
    2. A summary ``SystemMessage`` — injected by :func:`get_llm_context` only
       when a rolling summary exists, giving the model past context without
       re-sending every historical token.

    Args:
        state: Current graph state.

    Returns:
        Partial state dict with the new assistant message and, when a
        summarisation pass was triggered, the updated STM fields
        (``summary``, ``summarize_at``, ``context_start``).
    """
    total: int = len(state["messages"])
    # Use stored threshold or fall back to the module default for new sessions.
    summarize_at: int = state.get("summarize_at") or stm_config.threshold  # type: ignore[call-overload]

    stm_update: dict = {}
    if total >= summarize_at:
        # Plain model (no tools) keeps summarisation prompts clean.
        stm_update = summarize_and_trim(state, model)

    # Apply any pointer updates locally so get_llm_context sees the new window.
    effective_state: GraphState = {**state, **stm_update}  # type: ignore[typeddict-item]
    context = get_llm_context(effective_state)

    response = model_with_tools.invoke([_sys_msg] + context)
    return {"messages": [response], **stm_update}
