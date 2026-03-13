from fastapi import APIRouter, HTTPException, Depends, Request, WebSocket
from fastapi.responses import StreamingResponse
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
            session_id=request.session_id
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def stream_chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service)
) -> StreamingResponse:
    """Chat endpoint with SSE streaming for real-time token responses"""
    return StreamingResponse(
        service.chat_stream(
            message=request.message,
            conversation_history=request.conversation_history or [],
            session_id=request.session_id
        ),
        media_type="text/event-stream"
    )


@router.websocket("/chat/stream/ws")
async def stream_chat_websocket(websocket: WebSocket) -> None:
    """WebSocket endpoint for bidirectional chat streaming"""
    await websocket.accept()

    # Get service from app.state (websocket.app gives access to the FastAPI app)
    service = websocket.app.state.chat_service
    if service is None:
        await websocket.close(code=1011, reason="Services not initialized")
        return

    try:
        await service.chat_websocket(websocket)
    except Exception as e:
        # Send error before closing
        await websocket.send_json({"type": "error", "message": str(e)})
        await websocket.close(code=1011, reason="Internal error")
