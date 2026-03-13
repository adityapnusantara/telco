# tests/test_chat_service.py
import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from app.services.llm.chat import ChatService, ReplyClassification
from app.services.llm.agent import Agent
from app.services.llm.callbacks import CallbackHandler
from langchain_core.messages import AIMessage


def test_chat_service_init():
    """Test ChatService class initialization"""
    mock_agent = Mock(spec=Agent)
    mock_handler = Mock(spec=CallbackHandler)

    with patch('app.services.llm.chat.get_classification_prompt_obj') as mock_get_prompt, \
         patch('app.services.llm.chat.get_classification_config') as mock_get_config, \
         patch('app.services.llm.chat.ChatOpenAI') as mock_chat_openai, \
         patch('app.services.llm.chat.create_agent') as mock_create_agent:

        mock_prompt_obj = MagicMock()
        mock_prompt_obj.compile.return_value = "Compiled classification user prompt"
        mock_get_prompt.return_value = mock_prompt_obj
        mock_get_config.return_value = {"model": "gpt-4o", "temperature": 0}
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        mock_agent_instance = MagicMock()
        mock_create_agent.return_value = mock_agent_instance

        service = ChatService(agent=mock_agent, handler=mock_handler)

        assert service.agent == mock_agent
        assert service.handler == mock_handler
        assert service._classification_agent == mock_agent_instance
        assert service._classification_prompt_obj == mock_prompt_obj

        # Verify compile() was NOT called in __init__ (only called in _classify_reply)
        mock_prompt_obj.compile.assert_not_called()

        # Verify create_agent was called with static system prompt string (not the prompt object)
        mock_create_agent.assert_called_once()
        call_kwargs = mock_create_agent.call_args.kwargs
        assert isinstance(call_kwargs["system_prompt"], str)
        assert "customer service response classifier" in call_kwargs["system_prompt"].lower()
        assert call_kwargs["tools"] == []


@pytest.mark.asyncio
async def test_chat_service_chat_with_natural_text_response():
    """Test ChatService.chat method with natural text response from agent"""
    mock_agent = Mock()

    # Mock agent returning natural text (not structured JSON)
    mock_agent.invoke.return_value = {
        "messages": [AIMessage(content="You can check your balance in the MyTelco app.")]
    }

    mock_handler = Mock()
    mock_handler.handler = Mock()

    with patch('app.services.llm.chat.get_classification_prompt_obj') as mock_get_prompt, \
         patch('app.services.llm.chat.get_classification_config') as mock_get_config, \
         patch('app.services.llm.chat.ChatOpenAI') as mock_chat_openai, \
         patch('app.services.llm.chat.create_agent') as mock_create_agent:

        mock_prompt_obj = MagicMock()
        mock_prompt_obj.compile.return_value = "Compiled classification user prompt"
        mock_get_prompt.return_value = mock_prompt_obj
        mock_get_config.return_value = {"model": "gpt-4o", "temperature": 0}
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        mock_agent_instance = MagicMock()
        mock_create_agent.return_value = mock_agent_instance

        service = ChatService(agent=mock_agent, handler=mock_handler)

    # Mock the classification LLM
    service._classify_reply = AsyncMock(return_value=ReplyClassification(
        confidence_score=0.95,
        escalate=False
    ))

    result = await service.chat("How do I check my balance?", [], "conv-123")

    assert result.reply == "You can check your balance in the MyTelco app."
    assert result.escalate is False
    assert result.confidence_score == 0.95
    mock_agent.invoke.assert_called_once()


@pytest.mark.asyncio
async def test_chat_service_chat_with_escalation():
    """Test ChatService.chat method with escalation flag from classification"""
    mock_agent = Mock()

    mock_agent.invoke.return_value = {
        "messages": [AIMessage(content="I'm sorry, I cannot help with that. Please speak to a human.")]
    }

    mock_handler = Mock()
    mock_handler.handler = Mock()

    with patch('app.services.llm.chat.get_classification_prompt_obj') as mock_get_prompt, \
         patch('app.services.llm.chat.get_classification_config') as mock_get_config, \
         patch('app.services.llm.chat.ChatOpenAI') as mock_chat_openai, \
         patch('app.services.llm.chat.create_agent') as mock_create_agent:

        mock_prompt_obj = MagicMock()
        mock_prompt_obj.compile.return_value = "Compiled classification user prompt"
        mock_get_prompt.return_value = mock_prompt_obj
        mock_get_config.return_value = {"model": "gpt-4o", "temperature": 0}
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        mock_agent_instance = MagicMock()
        mock_create_agent.return_value = mock_agent_instance

        service = ChatService(agent=mock_agent, handler=mock_handler)

    # Mock classification returning escalate=True
    service._classify_reply = AsyncMock(return_value=ReplyClassification(
        confidence_score=0.5,
        escalate=True
    ))

    result = await service.chat("I want to sue you", [], "conv-456")

    assert result.escalate is True
    assert result.confidence_score == 0.5


@pytest.mark.asyncio
async def test_chat_service_chat_handles_empty_messages():
    """Test ChatService.chat handles empty messages list"""
    mock_agent = Mock()

    # Return empty messages list
    mock_agent.invoke.return_value = {
        "messages": []
    }

    mock_handler = Mock()
    mock_handler.handler = Mock()

    with patch('app.services.llm.chat.get_classification_prompt_obj') as mock_get_prompt, \
         patch('app.services.llm.chat.get_classification_config') as mock_get_config, \
         patch('app.services.llm.chat.ChatOpenAI') as mock_chat_openai, \
         patch('app.services.llm.chat.create_agent') as mock_create_agent:

        mock_prompt_obj = MagicMock()
        mock_prompt_obj.compile.return_value = "Compiled classification user prompt"
        mock_get_prompt.return_value = mock_prompt_obj
        mock_get_config.return_value = {"model": "gpt-4o", "temperature": 0}
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        mock_agent_instance = MagicMock()
        mock_create_agent.return_value = mock_agent_instance

        service = ChatService(agent=mock_agent, handler=mock_handler)

    result = await service.chat("Hello", [], "conv-789")

    # Should return defensive error response
    assert "couldn't generate a response" in result.reply.lower()
    assert result.escalate is True
    assert result.confidence_score == 0.0
    assert result.sources is None



def test_extract_sources_returns_none_when_empty():
    """Test _extract_sources returns None when no sources are found"""
    mock_agent = Mock()
    mock_handler = Mock()

    with patch('app.services.llm.chat.get_classification_prompt_obj') as mock_get_prompt, \
         patch('app.services.llm.chat.get_classification_config') as mock_get_config, \
         patch('app.services.llm.chat.ChatOpenAI') as mock_chat_openai, \
         patch('app.services.llm.chat.create_agent') as mock_create_agent:

        mock_prompt_obj = MagicMock()
        mock_prompt_obj.compile.return_value = "Compiled classification user prompt"
        mock_get_prompt.return_value = mock_prompt_obj
        mock_get_config.return_value = {"model": "gpt-4o", "temperature": 0}
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        mock_agent_instance = MagicMock()
        mock_create_agent.return_value = mock_agent_instance

        service = ChatService(agent=mock_agent, handler=mock_handler)
    result_dict = {"messages": []}

    sources = service._extract_sources(result_dict)

    assert sources is None


def test_extract_sources_extracts_from_tool_message():
    """Test _extract_sources extracts source names from ToolMessages"""
    from langchain_core.messages import ToolMessage

    mock_agent = Mock()
    mock_handler = Mock()

    with patch('app.services.llm.chat.get_classification_prompt_obj') as mock_get_prompt, \
         patch('app.services.llm.chat.get_classification_config') as mock_get_config, \
         patch('app.services.llm.chat.ChatOpenAI') as mock_chat_openai, \
         patch('app.services.llm.chat.create_agent') as mock_create_agent:

        mock_prompt_obj = MagicMock()
        mock_prompt_obj.compile.return_value = "Compiled classification user prompt"
        mock_get_prompt.return_value = mock_prompt_obj
        mock_get_config.return_value = {"model": "gpt-4o", "temperature": 0}
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        mock_agent_instance = MagicMock()
        mock_create_agent.return_value = mock_agent_instance

        service = ChatService(agent=mock_agent, handler=mock_handler)

    result_dict = {
        "messages": [
            ToolMessage(
                content="[billing - billing_qna.json]: Your bill is due on the 1st.",
                tool_call_id="test_tool_id"
            )
        ]
    }

    sources = service._extract_sources(result_dict)

    assert sources == ["billing_qna.json"]


def test_extract_sources_handles_multiple_sources():
    """Test _extract_sources handles multiple unique sources"""
    from langchain_core.messages import ToolMessage

    mock_agent = Mock()
    mock_handler = Mock()

    with patch('app.services.llm.chat.get_classification_prompt_obj') as mock_get_prompt, \
         patch('app.services.llm.chat.get_classification_config') as mock_get_config, \
         patch('app.services.llm.chat.ChatOpenAI') as mock_chat_openai, \
         patch('app.services.llm.chat.create_agent') as mock_create_agent:

        mock_prompt_obj = MagicMock()
        mock_prompt_obj.compile.return_value = "Compiled classification user prompt"
        mock_get_prompt.return_value = mock_prompt_obj
        mock_get_config.return_value = {"model": "gpt-4o", "temperature": 0}
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        mock_agent_instance = MagicMock()
        mock_create_agent.return_value = mock_agent_instance

        service = ChatService(agent=mock_agent, handler=mock_handler)

    result_dict = {
        "messages": [
            ToolMessage(
                content="[billing - billing_qna.json]: Bill info. [plans - plans_qna.json]: Plan info.",
                tool_call_id="test_tool_id"
            )
        ]
    }

    sources = service._extract_sources(result_dict)

    # Should extract both unique sources
    assert "billing_qna.json" in sources
    assert "plans_qna.json" in sources
