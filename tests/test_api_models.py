from pydantic import ValidationError
import pytest
from app.api.models import ChatRequest, ChatResponse

def test_chat_request_valid():
    """Test creating a valid chat request"""
    req = ChatRequest(message="What are your plans?", conversation_history=[], conversation_id=None)
    assert req.message == "What are your plans?"

def test_chat_request_minimal():
    """Test chat request with only required field"""
    req = ChatRequest(message="Test")
    assert req.message == "Test"

def test_chat_response_valid():
    """Test creating a valid chat response"""
    resp = ChatResponse(reply="Response", escalate=False)
    assert resp.reply == "Response"
    assert resp.escalate == False

def test_chat_request_empty_message():
    """Test that empty message raises validation error"""
    with pytest.raises(ValidationError):
        ChatRequest(message="")
