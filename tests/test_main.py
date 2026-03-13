from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


def test_app_exists():
    """Test that FastAPI app can be imported"""
    from app.main import app
    assert app is not None


def test_root_endpoint(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Telco Customer Service AI Agent"


def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@patch("app.services.rag.vector_store.QdrantVectorStore")
@patch("app.services.rag.vector_store.OpenAIEmbeddings")
@patch("app.services.rag.vector_store.QdrantClient")
@patch("app.services.llm.agent.ChatOpenAI")
@patch("app.services.llm.agent.get_agent_prompt")
@patch("app.services.llm.agent.create_agent")
@patch("app.services.llm.callbacks.LangfuseCallbackHandler")
def test_startup_creates_services(
    mock_langfuse_cb, mock_create_agent, mock_get_prompt,
    mock_chat_openai, mock_qdrant_client, mock_embeddings, mock_qdrant_vs
):
    """Test that startup event creates all services in app.state"""
    from app.main import app
    from app.services.rag.vector_store import VectorStore
    from app.services.rag.retriever import RetrieverTool
    from app.services.llm.callbacks import CallbackHandler
    from app.services.llm.agent import Agent
    from app.services.llm.chat import ChatService

    # Setup mocks
    mock_client_instance = MagicMock()
    mock_qdrant_client.return_value = mock_client_instance
    mock_embeddings_instance = MagicMock()
    mock_embeddings.return_value = mock_embeddings_instance
    mock_vs_instance = MagicMock()
    mock_qdrant_vs.return_value = mock_vs_instance
    mock_langfuse_handler_instance = MagicMock()
    mock_langfuse_cb.return_value = mock_langfuse_handler_instance
    mock_agent_instance = MagicMock()
    mock_create_agent.return_value = mock_agent_instance
    # get_agent_prompt returns a dict with system_prompt and model_config
    mock_get_prompt.return_value = {
        "system_prompt": [{"content": "You are a helpful assistant"}],
        "model_config": {"model": "gpt-4o", "temperature": 0}
    }

    with TestClient(app) as client:
        assert hasattr(app.state, 'vector_store')
        assert hasattr(app.state, 'retriever_tool')
        assert hasattr(app.state, 'callback_handler')
        assert hasattr(app.state, 'agent')
        assert hasattr(app.state, 'chat_service')
        assert isinstance(app.state.vector_store, VectorStore)
        assert isinstance(app.state.retriever_tool, RetrieverTool)
        assert isinstance(app.state.callback_handler, CallbackHandler)
        assert isinstance(app.state.agent, Agent)
        assert isinstance(app.state.chat_service, ChatService)
