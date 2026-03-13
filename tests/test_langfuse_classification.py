import pytest
from unittest.mock import patch, MagicMock
from app.prompts.langfuse import get_classification_prompt_obj, get_classification_prompt


def reset_singleton():
    """Reset the Langfuse client singleton between tests"""
    from app.prompts import langfuse
    langfuse._langfuse_client = None


def test_get_classification_prompt_obj():
    """Test getting classification prompt object from Langfuse"""
    reset_singleton()
    mock_prompt_obj = MagicMock()
    mock_prompt_obj.compile.return_value = "Compiled prompt"

    with patch('app.prompts.langfuse.get_client') as mock_get_client:
        mock_client = MagicMock()
        mock_client.get_prompt.return_value = mock_prompt_obj
        mock_get_client.return_value = mock_client

        result = get_classification_prompt_obj()

        mock_client.get_prompt.assert_called_once()
        assert result == mock_prompt_obj


def test_get_classification_prompt():
    """Test getting classification prompt and config from Langfuse"""
    reset_singleton()
    mock_prompt_obj = MagicMock()
    mock_prompt_obj.prompt = "You are a customer service response classifier."
    mock_prompt_obj.config = {"model": "gpt-4o", "temperature": 0}

    with patch('app.prompts.langfuse.get_client') as mock_get_client:
        mock_client = MagicMock()
        mock_client.get_prompt.return_value = mock_prompt_obj
        mock_get_client.return_value = mock_client

        result = get_classification_prompt()

        mock_client.get_prompt.assert_called_once()
        assert result == {
            "system_prompt": "You are a customer service response classifier.",
            "model_config": {"model": "gpt-4o", "temperature": 0}
        }
