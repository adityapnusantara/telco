import pytest
from unittest.mock import patch, MagicMock
from app.prompts.langfuse import get_classification_prompt


def reset_singleton():
    """Reset the Langfuse client singleton between tests"""
    from app.prompts import langfuse
    langfuse.get_langfuse_client.cache_clear()


def test_get_classification_prompt():
    """Test getting classification prompt and config from Langfuse"""
    reset_singleton()

    # Create a simple class to mock the prompt object with .prompt and .compile methods
    class MockPrompt:
        def __init__(self, prompt_text, config_dict):
            self.prompt = prompt_text
            self.config = config_dict

        def compile(self, **kwargs):
            return f"Compiled prompt with {kwargs}"

    config_dict = {"model": "gpt-4o", "temperature": 0}
    mock_system_prompt_obj = MockPrompt("You are a customer service response classifier.", config_dict)
    mock_user_prompt_obj = MockPrompt("", config_dict)

    with patch('app.prompts.langfuse.get_client') as mock_get_client:
        mock_client = MagicMock()
        # Set up the mock to return different prompts based on the name
        def get_prompt_side_effect(name):
            if "system" in name.lower():
                return mock_system_prompt_obj
            else:
                return mock_user_prompt_obj

        mock_client.get_prompt.side_effect = get_prompt_side_effect
        mock_get_client.return_value = mock_client

        result = get_classification_prompt()

        # Should call get_prompt twice (for system and user prompts)
        assert mock_client.get_prompt.call_count == 2
        # Check that user_prompt is the prompt object (has compile method)
        assert hasattr(result["user_prompt"], "compile")
        # Check model_config
        assert result["model_config"] == config_dict
