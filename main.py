"""Main API entry point for AgentWeave."""

from fastapi import FastAPI
from app.api import chat

app = FastAPI()

app.include_router(chat.router, prefix="/api/v1", tags=["chat"])

# uv run uvicorn app.main:app --reload
