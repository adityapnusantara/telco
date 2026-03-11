from unittest.mock import patch, MagicMock
from app.services.rag.vector_store import get_vector_store


@patch("app.services.rag.vector_store.QdrantVectorStore")
@patch("app.services.rag.vector_store.OpenAIEmbeddings")
@patch("app.services.rag.vector_store.QdrantClient")
def test_get_vector_store(mock_qdrant_client, mock_embeddings, mock_vector_store):
    """Test vector store initialization"""
    mock_client_instance = MagicMock()
    mock_qdrant_client.return_value = mock_client_instance

    mock_embeddings_instance = MagicMock()
    mock_embeddings.return_value = mock_embeddings_instance

    mock_vs_instance = MagicMock()
    mock_vector_store.return_value = mock_vs_instance

    vs = get_vector_store()

    assert vs is not None
    mock_vector_store.assert_called_once()
