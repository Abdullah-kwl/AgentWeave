"""Schemas for the API endpoints."""

from pydantic import BaseModel


class Messages(BaseModel):
    """Schema for incoming chat messages."""

    message: str
    session_id: str


class ChatResponse(BaseModel):
    """Schema for the chat response."""

    response: list[dict]
