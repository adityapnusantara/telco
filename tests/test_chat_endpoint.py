import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock
from app.services.llm.chat import ChatResponse


def test_chat_endpoint_with_depends():
    """Test chat endpoint using Depends injection"""
    # Create a test app without lifespan for unit testing
    from fastapi import FastAPI
    from app.api.routes.chat import router
    from app.api.models import ChatRequest

    test_app = FastAPI()
    test_app.include_router(router)

    mock_service = Mock()
    mock_service.chat.return_value = ChatResponse(
        reply="Test response",
        escalate=False
    )
    test_app.state.chat_service = mock_service

    with TestClient(test_app) as client:
        response = client.post("/chat", json={
            "message": "Hello",
            "conversation_history": [],
            "conversation_id": "test-123"
        })

        assert response.status_code == 200
        assert response.json()["reply"] == "Test response"
        assert response.json()["escalate"] is False


def test_chat_endpoint_returns_503_if_not_initialized():
    """Test that chat endpoint returns 503 if services not initialized"""
    from fastapi import FastAPI
    from app.api.routes.chat import router

    test_app = FastAPI()
    test_app.include_router(router)
    test_app.state.chat_service = None

    with TestClient(test_app) as client:
        response = client.post("/chat", json={"message": "Hello"})

        assert response.status_code == 503
        assert "not initialized" in response.json()["detail"]
