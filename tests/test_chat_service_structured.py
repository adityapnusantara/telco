import pytest
from unittest.mock import MagicMock, patch
from app.services.llm.chat import ChatService
from app.services.llm.agent import Agent
from app.services.llm.callbacks import CallbackHandler


def test_chat_service_extracts_structured_response():
    """Test that ChatService correctly extracts structured response"""
    # Mock agent
    mock_agent = MagicMock()
    mock_handler = MagicMock()

    # Create mock result with structured_response
    mock_result = {
        "messages": [],
        "structured_response": MagicMock(
            reply="Test reply",
            confidence_score=0.85,
            escalate=False
        )
    }
    mock_agent.invoke.return_value = mock_result

    # Create service
    service = ChatService(agent=mock_agent, handler=mock_handler)

    # Call chat
    response = service.chat(
        message="Test message",
        conversation_history=[],
        session_id="test-123"
    )

    # Assertions
    assert response.reply == "Test reply"
    assert response.confidence_score == 0.85
    assert response.escalate is False
    mock_agent.invoke.assert_called_once()


def test_chat_service_fallback_when_no_structured_response():
    """Test that ChatService uses fallback when structured_response is missing"""
    mock_agent = MagicMock()
    mock_handler = MagicMock()

    # Create mock result WITHOUT structured_response
    from langchain_core.messages import AIMessage
    mock_result = {
        "messages": [AIMessage(content="Fallback reply")]
    }
    mock_agent.invoke.return_value = mock_result

    service = ChatService(agent=mock_agent, handler=mock_handler)

    response = service.chat(
        message="Test message",
        conversation_history=[],
        session_id="test-123"
    )

    # Assertions - should use fallback
    assert response.reply == "Fallback reply"
    assert response.escalate is False  # Default in fallback
    assert response.confidence_score is None  # Not available in fallback


def test_chat_service_extracts_sources_from_tool_calls():
    """Test that _extract_sources correctly parses source names from tool results"""
    mock_agent = MagicMock()
    mock_handler = MagicMock()

    # Create mock result with tool results containing sources
    from langchain_core.messages import AIMessage, ToolMessage
    mock_result = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "search_knowledge_base",
                    "args": {"query": "test"},
                    "id": "test-id"
                }]
            ),
            ToolMessage(
                content="[Billing - billing_qna.json]: Billing info here\n[Plans - plans_qna.json]: Plans info here",
                tool_call_id="test-id"
            )
        ],
        "structured_response": MagicMock(
            reply="Test reply",
            confidence_score=0.9,
            escalate=False
        )
    }
    mock_agent.invoke.return_value = mock_result

    service = ChatService(agent=mock_agent, handler=mock_handler)

    response = service.chat(
        message="Test message",
        conversation_history=[],
        session_id="test-123"
    )

    # Should extract sources from tool results
    assert response.sources is not None
    assert "billing_qna.json" in response.sources
    assert "plans_qna.json" in response.sources
