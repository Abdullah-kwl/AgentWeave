"""APIRouter for chat endpoint it expose both streaning and none-streaming endpoints."""

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from app.api.schemas import Messages, ChatResponse, SessionId
from app.api.helper import chat_with_agent, chat_with_agent_stream, old_chat_history

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(message: Messages):
    """Endpoint for chatting with the agent. It returns the response in OpenAI format."""
    try:
        results = await chat_with_agent(message=message.message, session_id=message.session_id)
        return {"response": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/chat/stream")
async def stream_chat(message: Messages):
    """Endpoint for streaming chat responses from the agent in SSE format."""
    return StreamingResponse(
        chat_with_agent_stream(message.message, message.session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat/history")
async def get_chat_history(session_id: SessionId):
    """Endpoint for retrieving chat history for a given session."""
    try:
        history = await old_chat_history(session_id.session_id)
        return {"response": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
