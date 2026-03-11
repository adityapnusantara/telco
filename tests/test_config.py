import os
from dotenv import load_dotenv
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.core.config import config

def test_config_can_be_loaded():
    """Test that environment variables can be loaded"""
    load_dotenv()
    assert os.getenv("OPENAI_API_KEY") is not None or os.getenv("OPENAI_API_KEY", "") == "sk-proj-..."
    assert os.getenv("LANGFUSE_PUBLIC_KEY") is not None
    assert os.getenv("LANGFUSE_SECRET_KEY") is not None
    assert os.getenv("QDRANT_URL") is not None
    assert os.getenv("QDRANT_API_KEY") is not None


def test_config_class_attributes():
    """Test that Config class has all required attributes"""
    # Check OpenAI config
    assert hasattr(config, 'OPENAI_API_KEY')

    # Check Langfuse config
    assert hasattr(config, 'LANGFUSE_PUBLIC_KEY')
    assert hasattr(config, 'LANGFUSE_SECRET_KEY')
    assert hasattr(config, 'LANGFUSE_BASE_URL')

    # Check Qdrant config
    assert hasattr(config, 'QDRANT_URL')
    assert hasattr(config, 'QDRANT_API_KEY')
    assert hasattr(config, 'QDRANT_COLLECTION_NAME')

    # Check App config
    assert hasattr(config, 'APP_ENV')
    assert hasattr(config, 'LOG_LEVEL')


def test_config_default_values():
    """Test that Config class has correct default values"""
    assert config.QDRANT_COLLECTION_NAME == "telco_knowledge_base"
    assert config.APP_ENV in ["development", "production", "test"]
    assert config.LOG_LEVEL in ["debug", "info", "warning", "error", "critical"]
