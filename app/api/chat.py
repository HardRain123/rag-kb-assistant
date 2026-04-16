from fastapi import APIRouter
from pydantic import BaseModel
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(chat_request: ChatRequest):
    # Placeholder implementation - replace with actual chat logic
    return ChatResponse(answer=f"你刚刚说的是：{chat_request.message}")
