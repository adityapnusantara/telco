# tests/test_agent.py
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.services.llm.agent import Agent
from app.services.rag.vector_store import VectorStore


def test_agent_init():
    """Test Agent class initialization"""
    mock_vector_store = Mock(spec=VectorStore)
    with patch('app.services.llm.agent.ChatOpenAI') as mock_llm_class, \
         patch('app.services.llm.agent.get_system_prompt') as mock_prompt, \
         patch('app.services.llm.agent.get_retriever_tool') as mock_tool, \
         patch('app.services.llm.agent.create_agent') as mock_create_agent:
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        mock_prompt.return_value = [{"content": "You are a helpful assistant"}]
        mock_agent_instance = MagicMock()
        mock_create_agent.return_value = mock_agent_instance
        mock_tool.return_value = Mock()

        agent = Agent(vector_store=mock_vector_store)

        assert agent._vector_store == mock_vector_store
        assert agent._agent == mock_agent_instance

        # Verify create_agent was called with correct parameters
        mock_create_agent.assert_called_once()
        call_kwargs = mock_create_agent.call_args.kwargs
        assert "model" in call_kwargs
        assert "tools" in call_kwargs
        assert "system_prompt" in call_kwargs
        assert call_kwargs["system_prompt"] == "You are a helpful assistant"


def test_agent_invoke():
    """Test Agent invoke method"""
    mock_vector_store = Mock(spec=VectorStore)
    with patch('app.services.llm.agent.ChatOpenAI'), \
         patch('app.services.llm.agent.get_system_prompt') as mock_prompt, \
         patch('app.services.llm.agent.get_retriever_tool') as mock_tool, \
         patch('app.services.llm.agent.create_agent') as mock_create_agent:
        mock_prompt.return_value = [{"content": "You are a helpful assistant"}]
        mock_agent_instance = MagicMock()
        mock_create_agent.return_value = mock_agent_instance
        mock_tool.return_value = Mock()

        agent = Agent(vector_store=mock_vector_store)
        messages = [{"role": "user", "content": "Hello"}]
        config = {"config_key": "config_value"}
        mock_agent_instance.invoke.return_value = "Response"

        result = agent.invoke(messages, config)

        mock_agent_instance.invoke.assert_called_once_with(messages, config)
        assert result == "Response"


def test_get_agent_deprecated():
    """Test that get_agent() raises DeprecationWarning and NotImplementedError"""
    with pytest.warns(DeprecationWarning, match="Use Agent class instead"), \
         pytest.raises(NotImplementedError, match="Use Agent class instead"):
        from app.services.llm.agent import get_agent
        get_agent()
