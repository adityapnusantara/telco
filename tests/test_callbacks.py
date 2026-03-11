from unittest.mock import patch, MagicMock
from app.services.llm.callbacks import get_langfuse_handler


@patch('app.services.llm.callbacks.CallbackHandler')
def test_get_langfuse_handler(mock_callback_handler):
    """Test Langfuse callback handler creation"""
    mock_handler = MagicMock()
    mock_callback_handler.return_value = mock_handler

    handler = get_langfuse_handler()

    assert handler is not None
    mock_callback_handler.assert_called_once()
