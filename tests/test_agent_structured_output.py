# tests/test_agent_structured_output.py
import pytest
from app.services.llm.agent import Agent, StructuredChatResponse
from app.services.rag.vector_store import VectorStore


def test_structured_chat_response_model_exists():
    """Test that StructuredChatResponse model is defined"""
    from app.services.llm.agent import StructuredChatResponse
    assert hasattr(StructuredChatResponse, 'model_fields')
    assert 'reply' in StructuredChatResponse.model_fields
    assert 'confidence_score' in StructuredChatResponse.model_fields
    assert 'escalate' in StructuredChatResponse.model_fields


def test_structured_chat_response_confidence_validation():
    """Test that confidence_score must be between 0 and 1"""
    from app.services.llm.agent import StructuredChatResponse

    # Valid range
    valid = StructuredChatResponse(
        reply="Test response",
        confidence_score=0.8,
        escalate=False
    )
    assert valid.confidence_score == 0.8

    # Invalid: too high
    with pytest.raises(Exception):
        StructuredChatResponse(
            reply="Test response",
            confidence_score=1.5,
            escalate=False
        )

    # Invalid: too low
    with pytest.raises(Exception):
        StructuredChatResponse(
            reply="Test response",
            confidence_score=-0.1,
            escalate=False
        )


def test_agent_has_response_format_configured(monkeypatch):
    """Test that Agent is initialized with response_format"""
    # This test will fail initially, then pass after we update the code
    from unittest.mock import MagicMock, patch

    # Mock the dependencies
    mock_vector_store = MagicMock(spec=VectorStore)
    mock_llm = MagicMock()
    mock_system_prompt = "Test prompt"

    with patch('app.services.llm.agent.ChatOpenAI') as mock_chat_openai, \
         patch('app.services.llm.agent.get_system_prompt') as mock_get_prompt, \
         patch('app.services.llm.agent.create_agent') as mock_create_agent:

        mock_chat_openai.return_value = mock_llm
        mock_get_prompt.return_value = mock_system_prompt

        # Import after patching
        from app.services.llm.agent import Agent

        # Create agent
        agent = Agent(vector_store=mock_vector_store)

        # Verify create_agent was called with response_format
        mock_create_agent.assert_called_once()
        call_kwargs = mock_create_agent.call_args.kwargs
        assert 'response_format' in call_kwargs
        assert call_kwargs['response_format'] == StructuredChatResponse
