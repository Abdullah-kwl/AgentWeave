"""Frontend for AgentWeave using Streamlit."""

import uuid
import requests
import streamlit as st
from app.utils.config import app_config  # pylint: disable=no-name-in-module

# CONFIG
st.set_page_config(
    page_title="AgentWeave",
    page_icon="🤖",
    layout="centered",
)

FASTAPI_URL = app_config.fastapi_url  # pylint: disable=no-name-in-module
FASTAPI_STREAM_URL = app_config.fastapi_stream_url  # pylint: disable=no-name-in-module


# SESSION STATE
def init_session():
    """Initialize session state for messages and session ID."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "messages" not in st.session_state:
        st.session_state.messages = []


init_session()


# BACKEND CALL
def get_assistant_reply(message: str, session_id: str) -> str:
    """Call the backend API to get the assistant's reply for a given user message and session ID."""
    try:
        response = requests.post(
            FASTAPI_URL,
            json={
                "message": message,
                "session_id": session_id,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            return f"⚠️ Backend error: {data['error']}"

        assistant_messages = [m for m in data.get("response", []) if m.get("role") == "assistant"]

        return assistant_messages[-1]["content"] if assistant_messages else "No response."

    except requests.exceptions.ConnectionError:
        return "⚠️ Backend not reachable."
    except requests.exceptions.Timeout:
        return "⚠️ Request timed out."
    except Exception as e:  # pylint: disable=broad-exception-caught
        return f"⚠️ Unexpected error: {e}"


# BACKEND CALL FOR STREAMING
def stream_assistant_reply(message: str, session_id: str):
    """Call the backend streaming API to get the assistant's reply
    for a given user message and session ID, yielding chunks as they arrive."""
    try:
        with requests.post(
            FASTAPI_STREAM_URL,
            json={"message": message, "session_id": session_id},
            stream=True,
            timeout=60,
        ) as response:
            response.raise_for_status()
            buffer = ""
            for raw_chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                buffer += raw_chunk
                # SSE chunks are separated by \n\n
                while "\n\n" in buffer:
                    chunk, buffer = buffer.split("\n\n", 1)
                    for line in chunk.splitlines():
                        if line.startswith("data: "):
                            content = line[6:]
                            if content == "[DONE]":
                                return
                            yield content.replace("\\n", "\n")  # unescape here

    except requests.exceptions.ConnectionError:
        yield "⚠️ Backend not reachable."
    except requests.exceptions.Timeout:
        yield "⚠️ Request timed out."
    except Exception as e:  # pylint: disable=broad-exception-caught
        yield f"⚠️ Unexpected error: {e}"


# UI HEADER
st.title("🤖 AgentWeave")
st.caption(f"Session: `{st.session_state.session_id}`")
st.divider()


# RENDER CHAT
def render_messages():
    """Render the chat messages from the session state."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


render_messages()


# CHAT INPUT
prompt = st.chat_input("Type your message…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # For simple response
        # with st.spinner("Thinking..."):
        #     reply = get_assistant_reply(
        #         prompt,
        #         st.session_state.session_id
        #     )
        #     st.markdown(reply)

        # For streaming response
        reply = st.write_stream(stream_assistant_reply(prompt, st.session_state.session_id))

    st.session_state.messages.append({"role": "assistant", "content": reply})


# SIDEBAR
with st.sidebar:
    st.header("⚙️ Controls")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.markdown("**Backend:** `localhost:8000`")
    st.markdown("**Model:** LangChain Agent")

# uv run streamlit run frontend.py
