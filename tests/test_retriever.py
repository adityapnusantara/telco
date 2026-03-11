from unittest.mock import patch, MagicMock
from app.services.rag.retriever import get_retriever_tool, search_knowledge_base

@patch('app.services.rag.retriever.get_vector_store')
def test_get_retriever_tool(mock_get_vs):
    """Test creating retriever tool"""
    mock_vs = MagicMock()
    mock_vs.as_retriever.return_value = MagicMock()
    mock_get_vs.return_value = mock_vs

    tool = get_retriever_tool()

    assert tool is not None
    assert tool.name == "search_knowledge_base"
    assert "knowledge base" in tool.description.lower()


@patch('app.services.rag.retriever.get_vector_store')
def test_search_knowledge_base_with_results(mock_get_vs):
    """Test searching knowledge base with results"""
    # Setup mock
    mock_vs = MagicMock()
    mock_retriever = MagicMock()

    # Create mock documents
    mock_doc1 = MagicMock()
    mock_doc1.page_content = "5G offers faster speeds"
    mock_doc1.metadata = {"source": "technology.md", "category": "network"}

    mock_doc2 = MagicMock()
    mock_doc2.page_content = "Fiber plans start at $50"
    mock_doc2.metadata = {"source": "pricing.md", "category": "plans"}

    mock_retriever.invoke.return_value = [mock_doc1, mock_doc2]
    mock_vs.as_retriever.return_value = mock_retriever
    mock_get_vs.return_value = mock_vs

    # Test the function using invoke
    result = search_knowledge_base.invoke({"query": "5G speeds"})

    assert result is not None
    assert "5G offers faster speeds" in result
    assert "Fiber plans start at $50" in result
    assert "[network - technology.md]" in result
    assert "[plans - pricing.md]" in result
    mock_vs.as_retriever.assert_called_once_with(search_kwargs={"k": 3})
    mock_retriever.invoke.assert_called_once_with("5G speeds")


@patch('app.services.rag.retriever.get_vector_store')
def test_search_knowledge_base_empty_results(mock_get_vs):
    """Test searching knowledge base with no results"""
    # Setup mock
    mock_vs = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    mock_vs.as_retriever.return_value = mock_retriever
    mock_get_vs.return_value = mock_vs

    # Test the function using invoke
    result = search_knowledge_base.invoke({"query": "unknown query"})

    assert result == "No relevant information found in the knowledge base."
