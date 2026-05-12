"""This module initializes the language models used in the AI layer of the application."""

from langchain_groq import ChatGroq
from app.utils.config import secrets, config

model = ChatGroq(model=config.model, groq_api_key=secrets.GROQ_API_KEY)
