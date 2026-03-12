# tests/test_chat_service.py
import pytest
from unittest.mock import Mock, MagicMock, patch
from app.services.llm.chat import ChatService
from app.services.llm.agent import Agent
from app.services.llm.callbacks import CallbackHandler
from langchain_core.messages import AIMessage


def test_chat_service_init():
    """Test ChatService class initialization"""
    mock_agent = Mock(spec=Agent)
    mock_handler = Mock(spec=CallbackHandler)
    service = ChatService(agent=mock_agent, handler=mock_handler)
    assert service.agent == mock_agent
    assert service.handler == mock_handler


def test_chat_service_chat():
    """Test ChatService.chat method"""
    # Create mock message with content attribute
    mock_message = AIMessage(content="Hello! How can I help?")
    mock_agent = Mock()
    mock_agent.invoke.return_value = {
        "messages": [mock_message]
    }
    mock_handler = Mock()
    mock_handler.handler = Mock()

    service = ChatService(agent=mock_agent, handler=mock_handler)
    result = service.chat("Hello", [], "conv-123")
    assert result.reply == "Hello! How can I help?"
    assert result.escalate is False
    mock_agent.invoke.assert_called_once()
