from unittest.mock import patch, MagicMock
import pytest
from app.services.llm.callbacks import CallbackHandler, get_langfuse_handler


def test_callback_handler_init():
    """Test CallbackHandler class initialization"""
    handler = CallbackHandler()
    assert handler.handler is not None


@patch('app.services.llm.callbacks.LangfuseCallbackHandler')
def test_get_langfuse_handler_deprecated(mock_langfuse_callback_handler):
    """Test that the legacy get_langfuse_handler function raises DeprecationWarning"""
    with pytest.raises(DeprecationWarning):
        get_langfuse_handler()
