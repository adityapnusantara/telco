from fastapi import APIRouter, HTTPException
from app.api.models import ChatRequest, ChatResponse
from app.services.llm.chat import chat as chat_service

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def create_chat(request: ChatRequest) -> ChatResponse:
    """Chat endpoint for the Telco customer service agent"""
    try:
        response = chat_service(
            message=request.message,
            conversation_history=request.conversation_history or [],
            conversation_id=request.conversation_id
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
