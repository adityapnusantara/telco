from unittest.mock import patch, MagicMock
import pytest
from app.services.llm.callbacks import CallbackHandler


def test_callback_handler_init():
    """Test CallbackHandler class initialization"""
    handler = CallbackHandler()
    assert handler.handler is not None
