import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock
from app.api.routes.chat import router
from app.api.models import ChatRequest


def test_chat_stream_endpoint_sse_format():
    """POST /chat/stream should return SSE stream with token events"""
    # Create a test app
    from fastapi import FastAPI

    test_app = FastAPI()
    test_app.include_router(router)

    # Create mock service with async generator that accepts parameters
    async def mock_stream_generator(message, conversation_history, session_id=None):
        # Simulate streaming token events
        yield f"data: {json.dumps({'type': 'token', 'content': 'Hello'})}\n\n"
        yield f"data: {json.dumps({'type': 'token', 'content': ' world'})}\n\n"
        yield f"data: {json.dumps({'type': 'end', 'reply': 'Hello world', 'confidence_score': 0.8, 'escalate': False, 'sources': None})}\n\n"

    mock_service = Mock()
    mock_service.chat_stream = mock_stream_generator
    test_app.state.chat_service = mock_service

    with TestClient(test_app) as client:
        response = client.post(
            "/chat/stream",
            json={"message": "Hello", "conversation_history": [], "session_id": "test123"}
        )

        # Should return streaming response
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Parse SSE events
        events = []
        for line in response.text.split('\n'):
            if line.startswith("data: "):
                event_data = json.loads(line[6:])
                events.append(event_data)

        # Should have token events and end event
        assert len(events) > 0
        assert events[-1]["type"] == "end"


@pytest.mark.asyncio
async def test_chat_websocket_endpoint():
    """WS /chat/stream/ws should handle websocket communication"""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api.routes.chat import router

    # Create a test app
    test_app = FastAPI()
    test_app.include_router(router)

    # Create mock service with async websocket handler
    async def mock_chat_websocket(websocket):
        await websocket.send_json({"type": "token", "content": "Hello"})
        await websocket.send_json({"type": "token", "content": " world"})
        await websocket.send_json({
            "type": "end",
            "reply": "Hello world",
            "confidence_score": 0.8,
            "escalate": False,
            "sources": None
        })

    mock_service = Mock()
    mock_service.chat_websocket = mock_chat_websocket
    test_app.state.chat_service = mock_service

    with TestClient(test_app) as client:
        with client.websocket_connect("/chat/stream/ws") as websocket:
            # Send message
            websocket.send_json({
                "type": "message",
                "message": "Hello",
                "session_id": "test123",
                "conversation_history": []
            })

            # Receive events
            events = []
            while True:
                try:
                    event = websocket.receive_json()
                    events.append(event)
                    if event.get("type") == "end":
                        break
                except:
                    break

            # Should have received events
            assert len(events) > 0
            assert events[-1]["type"] == "end"
