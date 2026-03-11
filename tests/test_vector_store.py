from unittest.mock import patch, MagicMock
import pytest
from app.services.rag.vector_store import get_vector_store, VectorStore
from app.core.config import config


def test_get_vector_store_deprecated():
    """Test that get_vector_store raises DeprecationWarning and NotImplementedError"""
    with pytest.warns(DeprecationWarning, match="Use VectorStore class instead"):
        with pytest.raises(NotImplementedError, match="Use VectorStore class instead"):
            get_vector_store()


@patch("app.services.rag.vector_store.QdrantVectorStore")
@patch("app.services.rag.vector_store.OpenAIEmbeddings")
@patch("app.services.rag.vector_store.QdrantClient")
def test_vector_store_init(mock_qdrant_client, mock_embeddings, mock_vector_store):
    """Test VectorStore class initialization"""
    mock_client_instance = MagicMock()
    mock_qdrant_client.return_value = mock_client_instance

    mock_embeddings_instance = MagicMock()
    mock_embeddings.return_value = mock_embeddings_instance

    mock_vs_instance = MagicMock()
    mock_vector_store.return_value = mock_vs_instance

    store = VectorStore(
        qdrant_url=config.QDRANT_URL,
        qdrant_api_key=config.QDRANT_API_KEY,
        collection_name=config.QDRANT_COLLECTION_NAME
    )

    assert store.store is not None
    assert hasattr(store, '_client')
    assert hasattr(store, '_embeddings')

    # Verify the mocks were called correctly
    mock_qdrant_client.assert_called_once_with(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY
    )
    mock_embeddings.assert_called_once_with(
        model="text-embedding-3-small",
        dimensions=512
    )
    mock_vector_store.assert_called_once()
