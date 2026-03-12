from unittest.mock import patch, MagicMock, Mock
import pytest
from app.services.rag.retriever import get_retriever_tool, search_knowledge_base, RetrieverTool
from app.services.rag.vector_store import VectorStore


class TestRetrieverTool:
    """Tests for the new RetrieverTool class"""

    def test_retriever_tool_init(self):
        """Test RetrieverTool class initialization"""
        mock_store = Mock(spec=VectorStore)
        mock_store.store = Mock()
        tool = RetrieverTool(vector_store=mock_store)
        assert tool._vector_store == mock_store
        assert tool.tool.name == "search_knowledge_base"

    def test_retriever_tool_search_with_results(self):
        """Test RetrieverTool searching with results"""
        # Setup mock
        mock_store = Mock(spec=VectorStore)
        mock_store.store = Mock()
        mock_retriever = MagicMock()

        # Create mock documents
        mock_doc1 = MagicMock()
        mock_doc1.page_content = "5G offers faster speeds"
        mock_doc1.metadata = {"source": "technology.md", "category": "network"}

        mock_doc2 = MagicMock()
        mock_doc2.page_content = "Fiber plans start at $50"
        mock_doc2.metadata = {"source": "pricing.md", "category": "plans"}

        mock_retriever.invoke.return_value = [mock_doc1, mock_doc2]
        mock_store.store.as_retriever.return_value = mock_retriever

        # Create RetrieverTool and test
        retriever_tool = RetrieverTool(vector_store=mock_store)
        result = retriever_tool.tool.invoke({"query": "5G speeds"})

        assert result is not None
        assert "5G offers faster speeds" in result
        assert "Fiber plans start at $50" in result
        assert "[network - technology.md]" in result
        assert "[plans - pricing.md]" in result
        mock_store.store.as_retriever.assert_called_once_with(search_kwargs={"k": 3})
        mock_retriever.invoke.assert_called_once_with("5G speeds")

    def test_retriever_tool_search_empty_results(self):
        """Test RetrieverTool searching with no results"""
        # Setup mock
        mock_store = Mock(spec=VectorStore)
        mock_store.store = Mock()
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_store.store.as_retriever.return_value = mock_retriever

        # Create RetrieverTool and test
        retriever_tool = RetrieverTool(vector_store=mock_store)
        result = retriever_tool.tool.invoke({"query": "unknown query"})

        assert result == "No relevant information found in the knowledge base."

    def test_legacy_get_retriever_tool_raises_not_implemented(self):
        """Test legacy get_retriever_tool function raises NotImplementedError"""
        with pytest.raises(NotImplementedError):
            get_retriever_tool()

    def test_legacy_search_knowledge_base_raises_not_implemented(self):
        """Test legacy search_knowledge_base function raises NotImplementedError"""
        with pytest.raises(NotImplementedError):
            search_knowledge_base("test query")
