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


# ----------------------------------utilities--------------------------------------------------


def get_session_id():
    """Get the current session ID from session state."""
    session_id = str(uuid.uuid4())
    return session_id


def add_session_id(session_id: str):
    """Add a session ID to the list of all sessions."""
    if session_id not in st.session_state.all_sessions:
        st.session_state.all_sessions.append(session_id)


# SESSION STATE
def init_session():
    """Initialize session state for messages and session ID."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = get_session_id()

    if "all_sessions" not in st.session_state:
        st.session_state.all_sessions = []

    if "messages" not in st.session_state:
        st.session_state.messages = []


init_session()
add_session_id(st.session_state.session_id)


def reset_session():
    """Reset the session state for messages and session ID."""
    st.session_state.session_id = get_session_id()
    add_session_id(st.session_state.session_id)
    st.session_state.messages = []


# ----------------------------------Main Functionalities------------------------------------------


# BACKND CALL FOR HISTORY
def get_chat_history(session_id: str) -> list:
    """Fetch chat history for a given session ID from the backend."""
    try:
        response = requests.post(
            "http://127.0.0.1:8000/api/v1/chat/history",
            json={"session_id": session_id},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", [])
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Backend not reachable.")
    except requests.exceptions.Timeout:
        st.error("⚠️ Request timed out.")
    except Exception as e:  # pylint: disable=broad-exception-caught
        st.error(f"⚠️ Unexpected error: {e}")
    return []


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
    st.title("⚙️ Controls")

    if st.button("New chat", use_container_width=True):
        reset_session()
        st.rerun()

    if st.button("🗑️ Delete chat", use_container_width=True):
        current_id = st.session_state.session_id
        if current_id in st.session_state.all_sessions:
            st.session_state.all_sessions.remove(current_id)
        reset_session()  # creates new session_id and clears messages
        st.rerun()

    st.divider()
    st.header("My Conversations")
    # show all session IDs
    for s_id in st.session_state.all_sessions:
        is_active = s_id == st.session_state.session_id
        label = f"{'▶ ' if is_active else ''}{s_id[:8]}..."  # truncate for readability
        if st.button(label, key=s_id, use_container_width=True):
            history = get_chat_history(s_id)
            if history:
                st.session_state.session_id = s_id
                # map backend format to local messages format
                st.session_state.messages = history
                st.rerun()

# uv run streamlit run frontend.py
