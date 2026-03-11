from unittest.mock import patch, MagicMock
from app.services.llm.chat import chat, ChatResponse


@patch('app.services.llm.chat.get_langfuse_handler')
@patch('app.services.llm.chat.get_agent')
def test_chat_success(mock_get_agent, mock_get_handler):
    """Test successful chat response"""
    mock_agent = MagicMock()
    mock_response = MagicMock()
    mock_response.messages = [{"role": "assistant", "content": "Test response"}]
    mock_agent.invoke.return_value = mock_response
    mock_get_agent.return_value = mock_agent

    mock_handler = MagicMock()
    mock_get_handler.return_value = mock_handler

    response = chat("Hello", [], None)

    assert isinstance(response, ChatResponse)
    assert response.reply == "Test response"
    assert response.escalate == False


@patch('app.services.llm.chat.get_langfuse_handler')
@patch('app.services.llm.chat.get_agent')
def test_chat_with_escalation(mock_get_agent, mock_get_handler):
    """Test chat response that requires escalation"""
    mock_agent = MagicMock()
    mock_response = MagicMock()
    mock_response.messages = [{"role": "assistant", "content": "I cannot help with this issue. Please speak to a human agent."}]
    mock_agent.invoke.return_value = mock_response
    mock_get_agent.return_value = mock_agent

    mock_handler = MagicMock()
    mock_get_handler.return_value = mock_handler

    response = chat("Complex issue", [], None)

    assert isinstance(response, ChatResponse)
    assert response.reply == "I cannot help with this issue. Please speak to a human agent."
    assert response.escalate == True


@patch('app.services.llm.chat.get_langfuse_handler')
@patch('app.services.llm.chat.get_agent')
def test_chat_with_conversation_history(mock_get_agent, mock_get_handler):
    """Test chat with conversation history"""
    mock_agent = MagicMock()
    mock_response = MagicMock()
    mock_response.messages = [{"role": "assistant", "content": "Based on previous context..."}]
    mock_agent.invoke.return_value = mock_response
    mock_get_agent.return_value = mock_agent

    mock_handler = MagicMock()
    mock_get_handler.return_value = mock_handler

    history = [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"}
    ]
    response = chat("Follow up question", history, "conv-123")

    assert isinstance(response, ChatResponse)
    assert response.reply == "Based on previous context..."
    mock_agent.invoke.assert_called_once()
    call_args = mock_agent.invoke.call_args
    messages = call_args[0][0]["messages"]
    assert len(messages) == 3
    assert messages[0] == history[0]
    assert messages[1] == history[1]
    assert messages[2] == {"role": "user", "content": "Follow up question"}
