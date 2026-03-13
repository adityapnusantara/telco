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
            # Create mock message chunk that simulates AIMessage (no tool_call_id)
            message_chunk = Mock(spec=['content'])  # Only allow 'content' attribute
            message_chunk.content = chunk_text
            # Explicitly ensure tool_call_id doesn't exist (simulates AIMessage)
            del message_chunk.tool_call_id
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


@pytest.mark.asyncio
async def test_chat_websocket_yields_json_events():
    """ChatService.chat_websocket() should handle websocket communication"""
    # Arrange - minimal setup with mocks
    mock_agent = Mock()
    mock_handler = Mock()
    mock_handler.handler = Mock()

    service = ChatService(agent=mock_agent, handler=mock_handler)

    # Mock websocket
    websocket = MagicMock()
    websocket.receive_json = AsyncMock(return_value={
        "type": "message",
        "message": "Hello",
        "session_id": "test123",
        "conversation_history": []
    })
    websocket.send_json = AsyncMock()

    # Track if astream was called
    astream_called = []
    full_reply_chunks = ["Hello", " there", "!"]

    # Setup async generator for agent.astream
    async def mock_astream_generator(*args, **kwargs):
        astream_called.append(True)
        for chunk_text in full_reply_chunks:
            # Create mock message chunk that simulates AIMessage (no tool_call_id)
            message_chunk = Mock(spec=['content'])  # Only allow 'content' attribute
            message_chunk.content = chunk_text
            # Explicitly ensure tool_call_id doesn't exist (simulates AIMessage)
            del message_chunk.tool_call_id
            yield {"type": "messages", "ns": (), "data": (message_chunk, {})}

    mock_agent.astream = mock_astream_generator

    # Mock invoke for the final result
    mock_agent.invoke.return_value = {
        "messages": [AIMessage(content="Hello there!")]
    }

    # Act
    await service.chat_websocket(websocket)

    # Assert - should have sent events
    assert websocket.send_json.call_count > 0

    # First event should be token or end
    first_call = websocket.send_json.call_args_list[0][0][0]
    assert "type" in first_call
    assert first_call["type"] in ["token", "end"]

    # Last event should be end
    last_call = websocket.send_json.call_args_list[-1][0][0]
    assert last_call["type"] == "end"
    assert "reply" in last_call


@pytest.mark.asyncio
async def test_chat_stream_handles_errors():
    """ChatService.chat_stream() should handle errors gracefully"""
    # Arrange - mock agent that raises error
    mock_agent = Mock()
    mock_handler = Mock()
    mock_handler.handler = Mock()

    service = ChatService(agent=mock_agent, handler=mock_handler)

    # Mock astream to raise error (must be async generator)
    async def mock_astream_error(*args, **kwargs):
        raise Exception("LLM error")
        yield  # Make it an async generator

    mock_agent.astream = mock_astream_error

    # Act
    events = []
    async for event_str in service.chat_stream("Hello", [], "test"):
        event_data = json.loads(event_str[6:])
        events.append(event_data)
        # Should receive error event and stop
        if event_data.get("type") == "error":
            break

    # Assert - should have error event with generic message (security: no raw exceptions exposed)
    assert len(events) > 0
    assert events[0]["type"] == "error"
    assert events[0]["message"] == "An error occurred while processing your request. Please try again."
