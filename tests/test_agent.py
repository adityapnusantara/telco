# tests/test_agent.py
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from app.services.llm.agent import Agent
from app.services.rag.vector_store import VectorStore
from app.services.rag.retriever import RetrieverTool
from langchain_core.messages import HumanMessage


def test_agent_init():
    """Test Agent class initialization"""
    mock_vector_store = Mock(spec=VectorStore)
    mock_retriever_tool = Mock()
    mock_retriever_tool.tool = Mock()

    with patch('app.services.llm.agent.ChatOpenAI') as mock_llm_class, \
         patch('app.services.llm.agent.get_agent_prompt') as mock_prompt, \
         patch('app.services.llm.agent.create_agent') as mock_create_agent:
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        # get_agent_prompt returns a dict with system_prompt and model_config
        mock_prompt.return_value = {
            "system_prompt": [{"role": "system", "content": "You are a helpful assistant"}],
            "model_config": {"model": "gpt-4o", "temperature": 0}
        }
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
        assert call_kwargs["system_prompt"] == [{"role": "system", "content": "You are a helpful assistant"}]
        # response_format should NOT be present (removed for natural text output)
        assert "response_format" not in call_kwargs


def test_agent_invoke():
    """Test Agent invoke method"""
    mock_vector_store = Mock(spec=VectorStore)
    mock_retriever_tool = Mock()
    mock_retriever_tool.tool = Mock()

    with patch('app.services.llm.agent.ChatOpenAI'), \
         patch('app.services.llm.agent.get_agent_prompt') as mock_prompt, \
         patch('app.services.llm.agent.create_agent') as mock_create_agent:
        # get_agent_prompt returns a dict with system_prompt and model_config
        mock_prompt.return_value = {
            "system_prompt": [{"role": "system", "content": "You are a helpful assistant"}],
            "model_config": {"model": "gpt-4o", "temperature": 0}
        }
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
         patch('app.services.llm.agent.get_agent_prompt') as mock_prompt, \
         patch('app.services.llm.agent.create_agent') as mock_create_agent:
        # get_agent_prompt returns a dict with system_prompt and model_config
        mock_prompt.return_value = {
            "system_prompt": [{"role": "system", "content": "You are a helpful assistant"}],
            "model_config": {"model": "gpt-4o", "temperature": 0}
        }
        mock_agent_instance = MagicMock()
        mock_create_agent.return_value = mock_agent_instance

        agent = Agent(vector_store=mock_vector_store, retriever_tool=mock_retriever_tool)

        assert agent._vector_store == mock_vector_store
        assert agent._agent == mock_agent_instance
        # Verify the provided retriever tool was used
        call_kwargs = mock_create_agent.call_args.kwargs
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == [mock_retriever_tool.tool]


@pytest.mark.asyncio
async def test_agent_astream_yields_tokens():
    """Agent.astream() should yield token chunks"""
    # Arrange - minimal setup with mocks
    mock_vector_store = Mock(spec=VectorStore)
    mock_retriever_tool = Mock()
    mock_retriever_tool.tool = Mock()

    with patch('app.services.llm.agent.ChatOpenAI'), \
         patch('app.services.llm.agent.get_agent_prompt') as mock_prompt, \
         patch('app.services.llm.agent.create_agent') as mock_create_agent:
        # get_agent_prompt returns a dict with system_prompt and model_config
        mock_prompt.return_value = {
            "system_prompt": [{"role": "system", "content": "You are a helpful assistant"}],
            "model_config": {"model": "gpt-4o", "temperature": 0}
        }

        # Create mock agent instance with astream method
        mock_agent_instance = MagicMock()
        mock_create_agent.return_value = mock_agent_instance

        # Track if astream was called
        astream_called = []

        # Setup async generator for astream
        async def mock_astream_generator(*args, **kwargs):
            # Track that astream was called with correct args
            astream_called.append(True)
            # Simulate yielding chunks
            yield {"content": "Hello"}
            yield {"content": " world"}

        mock_agent_instance.astream = mock_astream_generator

        agent = Agent(vector_store=mock_vector_store, retriever_tool=mock_retriever_tool)
        messages = {"messages": [HumanMessage(content="Hello")]}

        # Act - collect streamed tokens
        tokens = []
        async for chunk in agent.astream(messages, config={}):
            tokens.append(chunk)

        # Assert - should receive at least one token
        assert len(tokens) > 0
        # Verify the mock agent's astream was called
        assert len(astream_called) > 0
