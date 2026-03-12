# tests/test_chat_service.py
import pytest
from unittest.mock import Mock, MagicMock, patch
from app.services.llm.chat import ChatService
from app.services.llm.agent import Agent, StructuredChatResponse
from app.services.llm.callbacks import CallbackHandler
from langchain_core.messages import AIMessage


def test_chat_service_init():
    """Test ChatService class initialization"""
    mock_agent = Mock(spec=Agent)
    mock_handler = Mock(spec=CallbackHandler)
    service = ChatService(agent=mock_agent, handler=mock_handler)
    assert service.agent == mock_agent
    assert service.handler == mock_handler


def test_chat_service_chat_with_structured_response():
    """Test ChatService.chat method with structured response from agent"""
    mock_agent = Mock()

    # Mock structured response
    mock_structured = StructuredChatResponse(
        reply="You can check your balance in the MyTelco app.",
        confidence_score=0.95,
        escalate=False
    )

    mock_agent.invoke.return_value = {
        "structured_response": mock_structured,
        "messages": [AIMessage(content="Test message")]
    }

    mock_handler = Mock()
    mock_handler.handler = Mock()

    service = ChatService(agent=mock_agent, handler=mock_handler)
    result = service.chat("How do I check my balance?", [], "conv-123")

    assert result.reply == "You can check your balance in the MyTelco app."
    assert result.escalate is False
    assert result.confidence_score == 0.95
    mock_agent.invoke.assert_called_once()


def test_chat_service_chat_with_escalation():
    """Test ChatService.chat method with escalation flag"""
    mock_agent = Mock()

    mock_structured = StructuredChatResponse(
        reply="I'm sorry, I cannot help with that. Please speak to a human.",
        confidence_score=0.5,
        escalate=True
    )

    mock_agent.invoke.return_value = {
        "structured_response": mock_structured,
        "messages": [AIMessage(content="Test message")]
    }

    mock_handler = Mock()
    mock_handler.handler = Mock()

    service = ChatService(agent=mock_agent, handler=mock_handler)
    result = service.chat("I want to sue you", [], "conv-456")

    assert result.escalate is True
    assert result.confidence_score == 0.5


def test_chat_service_chat_fallback_when_structured_missing():
    """Test ChatService.chat falls back when structured_response is missing"""
    mock_agent = Mock()

    # Return result without structured_response to trigger fallback
    mock_agent.invoke.return_value = {
        "messages": [AIMessage(content="I can help with basic questions.")]
    }

    mock_handler = Mock()
    mock_handler.handler = Mock()

    service = ChatService(agent=mock_agent, handler=mock_handler)
    result = service.chat("Hello", [], "conv-789")

    # Should use fallback response
    assert result.reply == "I can help with basic questions."
    assert result.escalate is False  # Fallback sets escalate to False
    assert result.confidence_score is None
    assert result.sources is None


def test_extract_sources_stub():
    """Test _extract_sources stub returns None (placeholder for Task 6)"""
    mock_agent = Mock()
    mock_handler = Mock()

    service = ChatService(agent=mock_agent, handler=mock_handler)
    result_dict = {"messages": []}

    sources = service._extract_sources(result_dict)

    # Stub returns None for now
    assert sources is None


def test_fallback_response_stub():
    """Test _fallback_response stub extracts reply from messages"""
    mock_agent = Mock()
    mock_handler = Mock()

    service = ChatService(agent=mock_agent, handler=mock_handler)
    result_dict = {
        "messages": [AIMessage(content="Fallback reply")]
    }

    response = service._fallback_response(result_dict)

    assert response.reply == "Fallback reply"
    assert response.escalate is False
    assert response.confidence_score is None
    assert response.sources is None
