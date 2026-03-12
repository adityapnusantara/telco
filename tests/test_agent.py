# tests/test_agent.py
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.services.llm.agent import Agent
from app.services.rag.vector_store import VectorStore


def test_agent_init():
    """Test Agent class initialization"""
    mock_vector_store = Mock(spec=VectorStore)
    mock_retriever_tool = Mock()
    mock_retriever_tool.tool = Mock()

    with patch('app.services.llm.agent.ChatOpenAI') as mock_llm_class, \
         patch('app.services.llm.agent.get_system_prompt') as mock_prompt, \
         patch('app.services.llm.agent.create_agent') as mock_create_agent:
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        # get_system_prompt returns a compiled string, not list of dict
        mock_prompt.return_value = "You are a helpful assistant"
        mock_agent_instance = MagicMock()
        mock_create_agent.return_value = mock_agent_instance

        agent = Agent(vector_store=mock_vector_store, retriever_tool=mock_retriever_tool)

        assert agent._vector_store == mock_vector_store
        assert agent._agent == mock_agent_instance

        # Verify create_agent was called with correct parameters
        mock_create_agent.assert_called_once()
        call_kwargs = mock_create_agent.call_args.kwargs
        assert "model" in call_kwargs
        assert "tools" in call_kwargs
        assert "system_prompt" in call_kwargs
        assert call_kwargs["system_prompt"] == "You are a helpful assistant"
        assert "response_format" in call_kwargs


def test_agent_invoke():
    """Test Agent invoke method"""
    mock_vector_store = Mock(spec=VectorStore)
    mock_retriever_tool = Mock()
    mock_retriever_tool.tool = Mock()

    with patch('app.services.llm.agent.ChatOpenAI'), \
         patch('app.services.llm.agent.get_system_prompt') as mock_prompt, \
         patch('app.services.llm.agent.create_agent') as mock_create_agent:
        # get_system_prompt returns a compiled string
        mock_prompt.return_value = "You are a helpful assistant"
        mock_agent_instance = MagicMock()
        mock_create_agent.return_value = mock_agent_instance

        agent = Agent(vector_store=mock_vector_store, retriever_tool=mock_retriever_tool)
        messages = [{"role": "user", "content": "Hello"}]
        config = {"config_key": "config_value"}
        mock_agent_instance.invoke.return_value = "Response"

        result = agent.invoke(messages, config)

        mock_agent_instance.invoke.assert_called_once_with(messages, config)
        assert result == "Response"


def test_agent_with_retriever_tool():
    """Test Agent initialization with provided RetrieverTool"""
    mock_vector_store = Mock(spec=VectorStore)
    mock_retriever_tool = Mock()
    mock_retriever_tool.tool = Mock()

    with patch('app.services.llm.agent.ChatOpenAI'), \
         patch('app.services.llm.agent.get_system_prompt') as mock_prompt, \
         patch('app.services.llm.agent.create_agent') as mock_create_agent:
        # get_system_prompt returns a compiled string
        mock_prompt.return_value = "You are a helpful assistant"
        mock_agent_instance = MagicMock()
        mock_create_agent.return_value = mock_agent_instance

        agent = Agent(vector_store=mock_vector_store, retriever_tool=mock_retriever_tool)

        assert agent._vector_store == mock_vector_store
        assert agent._agent == mock_agent_instance
        # Verify the provided retriever tool was used
        call_kwargs = mock_create_agent.call_args.kwargs
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == [mock_retriever_tool.tool]
