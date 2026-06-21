"""This module initializes the language models used in the AI layer of the application."""

from langchain_groq import ChatGroq
from app.utils.config import secrets, llm_config


def initialize_model():
    """Initializes the ChatGroq model with the provided API key."""
    return ChatGroq(model=llm_config.model, groq_api_key=secrets.GROQ_API_KEY)


def initialize_tool_enabled_model(model, tools):
    """Binds the provided tools to the model."""
    return model.bind_tools(tools)
