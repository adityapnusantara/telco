from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from app.core.config import config


class VectorStore:
    """Qdrant vector store wrapper with explicit initialization"""

    def __init__(self, qdrant_url: str, qdrant_api_key: str, collection_name: str):
        self._client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self._collection_name = collection_name
        self._embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            dimensions=config.EMBEDDING_DIMENSION
        )

        # Create collection if it doesn't exist
        self._ensure_collection_exists()

        self._store = QdrantVectorStore(
            client=self._client,
            collection_name=collection_name,
            embedding=self._embeddings
        )

    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist"""
        from qdrant_client.http.exceptions import UnexpectedResponse

        try:
            self._client.get_collection(collection_name=self._collection_name)
        except (UnexpectedResponse, Exception):
            # Collection doesn't exist, create it
            print(f"Creating collection: {self._collection_name}")
            print(f"Embedding model: {config.EMBEDDING_MODEL}")
            print(f"Embedding dimension: {config.EMBEDDING_DIMENSION}")
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(size=config.EMBEDDING_DIMENSION, distance=Distance.COSINE)
            )
            print(f"✅ Collection created successfully")

    @property
    def store(self):
        """Get the underlying LangChain vector store"""
        return self._store
