import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Langfuse
    LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    LANGFUSE_BASE_URL: str = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

    # Langfuse Prompt Names
    SYSTEM_PROMPT_NAME: str = os.getenv("SYSTEM_PROMPT_NAME", "telco-customer-service-agent")
    CLASSIFICATION_SYSTEM_PROMPT_NAME: str = os.getenv("CLASSIFICATION_SYSTEM_PROMPT_NAME", "telco-customer-service-classification-user")
    CLASSIFICATION_USER_PROMPT_NAME: str = os.getenv("CLASSIFICATION_USER_PROMPT_NAME", "telco-customer-service-classification-system")

    # Qdrant
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "telco_knowledge_base")

    # Embeddings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "1536"))

    # App
    APP_ENV: str = os.getenv("APP_ENV", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")

    # LLM Defaults
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gpt-4o")
    DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0"))

config = Config()
