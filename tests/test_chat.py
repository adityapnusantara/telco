# tests/test_chat.py
import pytest
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from app.services.llm.chat import ChatService
from app.services.llm.agent import Agent
from app.services.llm.callbacks import CallbackHandler
from app.services.rag.vector_store import VectorStore
from app.services.rag.retriever import RetrieverTool
from langchain_core.messages import AIMessage, HumanMessage


@pytest.mark.asyncio
async def test_chat_stream_yields_sse_events():
    """ChatService.chat_stream() should yield SSE-formatted events"""
    # Arrange - minimal setup with mocks
    mock_agent = Mock()
    mock_handler = Mock()
    mock_handler.handler = Mock()

    service = ChatService(agent=mock_agent, handler=mock_handler)

    # Track if astream was called
    astream_called = []
    full_reply_chunks = ["Hello", " there", "!"]

    # Setup async generator for agent.astream
    async def mock_astream_generator(*args, **kwargs):
        astream_called.append(True)
        # Simulate yielding chunks with message_chunk objects
        # Actual format from LangChain astream with stream_mode="messages" is:
        # {'type': 'messages', 'ns': (), 'data': (AIMessageChunk, metadata)}
        for chunk_text in full_reply_chunks:
            # Create mock message chunk with content attribute (not text)
            message_chunk = Mock()
            message_chunk.content = chunk_text
            yield {"type": "messages", "ns": (), "data": (message_chunk, {})}

    mock_agent.astream = mock_astream_generator

    # Mock invoke for the final result (used to extract sources)
    mock_agent.invoke.return_value = {
        "messages": [AIMessage(content="Hello there!")]
    }

    # Act - collect SSE events
    events = []
    async for event_str in service.chat_stream(
        message="Hello",
        conversation_history=[],
        session_id="test123"
    ):
        # Parse SSE format: "data: {...}\n\n"
        assert event_str.startswith("data: ")
        event_data = json.loads(event_str[6:])  # Remove "data: " prefix
        events.append(event_data)

    # Assert - should have token events and final end event
    assert len(events) > 0
    # First event should be a token event
    assert events[0]["type"] == "token"
    assert events[0]["content"] == "Hello"
    # Last event should be end event
    assert events[-1]["type"] == "end"
    assert "reply" in events[-1]
    assert "confidence_score" in events[-1]
    assert "escalate" in events[-1]
    # Verify astream was called
    assert len(astream_called) > 0
