from unittest.mock import patch, MagicMock
from app.prompts import langfuse

def reset_singleton():
    """Reset the Langfuse client singleton between tests"""
    langfuse._langfuse_client = None

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

    # Set up the compile method to return our expected result
    expected_prompt = [{"role": "system", "content": "You are a helpful assistant."}]
    mock_prompt.compile.return_value = expected_prompt

    mock_client.get_prompt.return_value = mock_prompt
    mock_get_client.return_value = mock_client

    prompt = langfuse.get_system_prompt("telco-customer-service-agent")

    assert prompt == expected_prompt
    mock_client.get_prompt.assert_called_once_with(
        "telco-customer-service-agent",
        type="chat"
    )
    # Verify compile was called with the default values
    mock_prompt.compile.assert_called_once_with(
        company_name="MyTelco",
        escalation_contact="call 123 or use the MyTelco app"
    )
