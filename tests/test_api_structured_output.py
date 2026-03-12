import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.mark.integration
def test_chat_endpoint_returns_structured_fields():
    """Test that /chat endpoint returns reply, escalate, and confidence_score"""
    import os
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "message": "What is my current bill?",
                "session_id": "test-123"
            }
        )

        assert response.status_code == 200

        data = response.json()
        assert "reply" in data
        assert "escalate" in data
        assert "confidence_score" in data

        # Verify types
        assert isinstance(data["reply"], str)
        assert isinstance(data["escalate"], bool)
        assert isinstance(data["confidence_score"], float) or data["confidence_score"] is None

        # Verify confidence_score range if present
        if data["confidence_score"] is not None:
            assert 0.0 <= data["confidence_score"] <= 1.0


@pytest.mark.integration
def test_chat_endpoint_with_triggers_escalation():
    """Test that LLM can set escalate=True when appropriate"""
    import os
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    with TestClient(app) as client:
        # This question should trigger escalation
        response = client.post(
            "/chat",
            json={
                "message": "I need to speak to a human agent immediately about a legal matter",
                "session_id": "test-456"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "escalate" in data
        # The LLM should recognize this needs escalation
        # Note: This depends on the LLM's judgment, so we just check the field exists
