from fastapi import APIRouter, HTTPException, Depends, Request
from app.api.models import ChatRequest, ChatResponse
from app.services.llm.chat import ChatService

router = APIRouter()


def get_chat_service(request: Request) -> ChatService:
    """Dependency to get ChatService from app.state"""
    service = request.app.state.chat_service
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Services not initialized"
        )
    return service


@router.post("/chat", response_model=ChatResponse)
async def create_chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service)
) -> ChatResponse:
    """Chat endpoint for the Telco customer service agent"""
    try:
        response = service.chat(
            message=request.message,
            conversation_history=request.conversation_history or [],
            conversation_id=request.conversation_id
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
