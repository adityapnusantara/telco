from unittest.mock import patch, MagicMock
from app.prompts import langfuse
from app.core.config import config

def reset_singleton():
    """Reset Langfuse client singleton between tests"""
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
def test_get_agent_prompt(mock_get_client):
    """Test fetching agent prompt and model config from Langfuse in one call"""
    reset_singleton()
    mock_client = MagicMock()
    mock_prompt = MagicMock()

    # Set up get_langchain_prompt to return our expected result
    expected_prompt = [{"role": "system", "content": "You are a helpful assistant."}]
    mock_prompt.get_langchain_prompt.return_value = expected_prompt

    # Mock model config
    mock_prompt.config = {
        "model": "gpt-4o",
        "temperature": 0
    }

    mock_client.get_prompt.return_value = mock_prompt
    mock_get_client.return_value = mock_client

    result = langfuse.get_agent_prompt()

    assert result["system_prompt"] == expected_prompt
    assert result["model_config"]["model"] == "gpt-4o"
    assert result["model_config"]["temperature"] == 0
    mock_client.get_prompt.assert_called_once_with(config.AGENT_PROMPT_NAME)
    # Verify get_langchain_prompt was called
    mock_prompt.get_langchain_prompt.assert_called_once()
    # Verify compile was not called (variables are hardcoded in Langfuse)
    mock_prompt.compile.assert_not_called()


@patch('app.prompts.langfuse.get_client')
def test_get_extraction_prompt(mock_get_client):
    """Test fetching extraction prompts and model config from Langfuse."""
    reset_singleton()
    mock_client = MagicMock()
    mock_system_prompt = MagicMock()
    mock_user_prompt = MagicMock()

    mock_system_prompt.prompt = "You are an extraction assistant."
    mock_system_prompt.config = {
        "model": "gpt-4o-mini",
        "temperature": 0
    }
    mock_client.get_prompt.side_effect = [mock_system_prompt, mock_user_prompt]
    mock_get_client.return_value = mock_client

    result = langfuse.get_extraction_prompt()

    assert result["system_prompt"] == "You are an extraction assistant."
    assert result["user_prompt"] == mock_user_prompt
    assert result["model_config"]["model"] == "gpt-4o-mini"
    assert result["model_config"]["temperature"] == 0
    assert mock_client.get_prompt.call_count == 2
    mock_client.get_prompt.assert_any_call(config.EXTRACTION_SYSTEM_PROMPT_NAME)
    mock_client.get_prompt.assert_any_call(config.EXTRACTION_USER_PROMPT_NAME)
