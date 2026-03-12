# tests/test_agent_integration_structured.py
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.tools import tool
from app.services.llm.agent import Agent, StructuredChatResponse
from app.services.rag.vector_store import VectorStore


@pytest.mark.integration
def test_agent_invoke_returns_structured_response():
    """Test that agent invoke returns structured_response in result"""
    # This is an integration test - it will call the actual OpenAI API
    # Skip if OPENAI_API_KEY is not set
    import os
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    # Create real vector store (or use mock for faster tests)
    mock_vector_store = MagicMock(spec=VectorStore)

    # Create a mock retriever tool with .tool property to mimic RetrieverTool
    @tool
    def mock_search_knowledge_base(query: str) -> str:
        """Mock search tool for testing"""
        return "No relevant information found in the knowledge base."

    # Create a mock RetrieverTool-like object with .tool property
    mock_retriever_tool = MagicMock()
    mock_retriever_tool.tool = mock_search_knowledge_base

    with patch('app.services.llm.agent.get_system_prompt') as mock_get_prompt:
        mock_get_prompt.return_value = "You are a helpful assistant."

        agent = Agent(vector_store=mock_vector_store, retriever_tool=mock_retriever_tool)

        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Hello"}]},
            {}
        )

    # Assertions
    assert "structured_response" in result
    assert hasattr(result["structured_response"], "reply")
    assert hasattr(result["structured_response"], "confidence_score")
    assert hasattr(result["structured_response"], "escalate")
    assert isinstance(result["structured_response"].confidence_score, float)
    assert 0.0 <= result["structured_response"].confidence_score <= 1.0
    assert isinstance(result["structured_response"].escalate, bool)
