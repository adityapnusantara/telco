from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from app.core.config import config

# Singleton instances
_qdrant_client = None
_vector_store = None


def get_qdrant_client():
    """Get or create Qdrant client singleton"""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
        )
    return _qdrant_client


def get_vector_store():
    """
    Get or create Qdrant vector store singleton.

    Returns:
        QdrantVectorStore instance
    """
    global _vector_store
    if _vector_store is None:
        client = get_qdrant_client()
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=512,
        )

        _vector_store = QdrantVectorStore(
            client=client,
            collection_name=config.QDRANT_COLLECTION_NAME,
            embedding=embeddings,
        )
    return _vector_store
