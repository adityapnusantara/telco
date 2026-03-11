import warnings
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from app.core.config import config


class VectorStore:
    """Qdrant vector store wrapper with explicit initialization"""

    def __init__(self, qdrant_url: str, qdrant_api_key: str, collection_name: str):
        self._client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)
        self._store = QdrantVectorStore(
            client=self._client,
            collection_name=collection_name,
            embedding=self._embeddings
        )

    @property
    def store(self):
        """Get the underlying LangChain vector store"""
        return self._store


# Singleton instances (deprecated, kept for backward compatibility)
_qdrant_client = None
_vector_store = None


# Legacy function for backward compatibility (will be removed after migration)
def get_qdrant_client():
    """Deprecated: Use VectorStore class instead"""
    warnings.warn("Use VectorStore class instead", DeprecationWarning, stacklevel=2)
    raise NotImplementedError("Use VectorStore class instead")


def get_vector_store():
    """Deprecated: Use VectorStore class instead"""
    warnings.warn("Use VectorStore class instead", DeprecationWarning, stacklevel=2)
    raise NotImplementedError("Use VectorStore class instead")
