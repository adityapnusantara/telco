from unittest.mock import patch, MagicMock
from app.prompts import langfuse
from app.core.config import config

def reset_singleton():
    """Reset the Langfuse client singleton between tests"""
    langfuse.get_langfuse_client.cache_clear()

def test_get_langfuse_client():
    """Test Langfuse client initialization"""
    reset_singleton()
    with patch('app.prompts.langfuse.get_client') as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        client = langfuse.get_langfuse_client()

        assert client is not None
        mock_get_client.assert_called_once()

@patch('app.prompts.langfuse.get_client')
def test_get_system_prompt(mock_get_client):
    """Test fetching system prompt from Langfuse"""
    reset_singleton()
    mock_client = MagicMock()
    mock_prompt = MagicMock()

    # Set up get_langchain_prompt to return our expected result
    expected_prompt = [{"role": "system", "content": "You are a helpful assistant."}]
    mock_prompt.get_langchain_prompt.return_value = expected_prompt

    mock_client.get_prompt.return_value = mock_prompt
    mock_get_client.return_value = mock_client

    prompt = langfuse.get_system_prompt()

    assert prompt == expected_prompt
    mock_client.get_prompt.assert_called_once_with(
        config.SYSTEM_PROMPT_NAME,
        type="chat"
    )
    # Verify get_langchain_prompt was called
    mock_prompt.get_langchain_prompt.assert_called_once()
    # Verify compile was not called (variables are hardcoded in Langfuse)
    mock_prompt.compile.assert_not_called()

@patch('app.prompts.langfuse.get_client')
def test_get_model_config(mock_get_client):
    """Test fetching model config from Langfuse"""
    reset_singleton()
    mock_client = MagicMock()
    mock_prompt = MagicMock()

    # Mock config with model and temperature
    mock_prompt.config = {
        "model": "gpt-4o",
        "temperature": 0
    }

    mock_client.get_prompt.return_value = mock_prompt
    mock_get_client.return_value = mock_client

    model_config = langfuse.get_model_config()

    assert model_config["model"] == "gpt-4o"
    assert model_config["temperature"] == 0
    mock_client.get_prompt.assert_called_once_with(
        config.SYSTEM_PROMPT_NAME,
        type="chat"
    )

@patch('app.prompts.langfuse.get_client')
def test_get_model_config_with_fallback(mock_get_client):
    """Test model config uses fallback when Langfuse config is empty"""
    reset_singleton()
    mock_client = MagicMock()
    mock_prompt = MagicMock()

    # Mock empty config
    mock_prompt.config = {}

    mock_client.get_prompt.return_value = mock_prompt
    mock_get_client.return_value = mock_client

    model_config = langfuse.get_model_config()

    # Should use fallback values from config.py
    assert model_config["model"] == config.DEFAULT_MODEL
    assert model_config["temperature"] == config.DEFAULT_TEMPERATURE
