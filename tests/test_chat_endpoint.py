from fastapi.testclient import TestClient
from unittest.mock import patch

def test_chat_endpoint_success(client):
    """Test successful chat request"""
    with patch('app.api.routes.chat.chat_service') as mock_chat:
        from app.services.llm.chat import ChatResponse
        mock_chat.return_value = ChatResponse(reply="Test response", escalate=False)

        response = client.post("/chat", json={"message": "What are your plans?", "conversation_history": []})

        assert response.status_code == 200
        data = response.json()
        assert data["reply"] == "Test response"
        assert data["escalate"] == False
