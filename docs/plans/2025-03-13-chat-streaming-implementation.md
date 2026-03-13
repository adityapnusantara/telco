# Chat Streaming Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add SSE and WebSocket streaming endpoints for real-time token-by-token chat responses.

**Architecture:** Extend existing Agent and ChatService classes with async streaming methods. Use FastAPI's StreamingResponse for SSE and native WebSocket support. LangChain's astream(stream_mode="messages") provides token stream.

**Tech Stack:** FastAPI (StreamingResponse, WebSocket), LangChain (astream), Server-Sent Events (SSE), httpx (testing), websockets (testing)

---

## Task 1: Add Agent.astream() method

**Files:**
- Modify: `app/services/llm/agent.py`
- Test: `tests/test_agent.py` (create if not exists)

**Step 1: Write the failing test**

Create `tests/test_agent.py`:

```python
import pytest
import asyncio
from app.services.llm.agent import Agent
from app.services.rag.vector_store import VectorStore
from app.services.rag.retriever import RetrieverTool
from langchain_core.messages import HumanMessage

@pytest.mark.asyncio
async def test_agent_astream_yields_tokens():
    """Agent.astream() should yield token chunks"""
    # Arrange - minimal setup
    vector_store = VectorStore()  # Uses env vars
    retriever_tool = RetrieverTool(vector_store=vector_store)
    agent = Agent(vector_store=vector_store, retriever_tool=retriever_tool)

    messages = {"messages": [HumanMessage(content="Hello")]}

    # Act - collect streamed tokens
    tokens = []
    async for chunk in agent.astream(messages, config={}):
        tokens.append(chunk)

    # Assert - should receive at least one token
    assert len(tokens) > 0
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_agent.py::test_agent_astream_yields_tokens -v`

Expected: `AttributeError: 'Agent' object has no attribute 'astream'`

**Step 3: Write minimal implementation**

Add to `app/services/llm/agent.py` after the `invoke()` method:

```python
async def astream(self, messages, config):
    """Stream agent responses token-by-token.

    Yields chunks from LangChain's astream with stream_mode="messages".
    """
    async for chunk in self._agent.astream(
        messages,
        config,
        stream_mode="messages",
        version="v2"
    ):
        yield chunk
```

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_agent.py::test_agent_astream_yields_tokens -v`

Expected: PASS

**Step 5: Commit**

```bash
git add app/services/llm/agent.py tests/test_agent.py
git commit -m "feat: add Agent.astream() method for token streaming"
```

---

## Task 2: Add ChatService.chat_stream() SSE generator

**Files:**
- Modify: `app/services/llm/chat.py`
- Test: `tests/test_chat.py` (create if not exists)

**Step 1: Write the failing test**

Create `tests/test_chat.py`:

```python
import pytest
import json
from app.services.llm.chat import ChatService
from app.services.llm.agent import Agent
from app.services.llm.callbacks import CallbackHandler
from app.services.rag.vector_store import VectorStore
from app.services.rag.retriever import RetrieverTool

@pytest.mark.asyncio
async def test_chat_stream_yields_sse_events():
    """ChatService.chat_stream() should yield SSE-formatted events"""
    # Arrange
    vector_store = VectorStore()
    retriever_tool = RetrieverTool(vector_store=vector_store)
    agent = Agent(vector_store=vector_store, retriever_tool=retriever_tool)
    handler = CallbackHandler()
    service = ChatService(agent=agent, handler=handler)

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
    assert events[0]["type"] in ["token", "end"]  # First event
    assert events[-1]["type"] == "end"  # Last event
    assert "reply" in events[-1]
    assert "confidence_score" in events[-1]
    assert "escalate" in events[-1]
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_chat.py::test_chat_stream_yields_sse_events -v`

Expected: `AttributeError: 'ChatService' object has no attribute 'chat_stream'`

**Step 3: Write minimal implementation**

Add to `app/services/llm/chat.py` after the `chat()` method:

```python
async def chat_stream(self, message: str, conversation_history: list[dict], session_id: Optional[str] = None):
    """Stream chat response via Server-Sent Events.

    Yields SSE-formatted strings:
    - Token events: "data: {"type": "token", "content": "..."}\n\n"
    - End event: "data: {"type": "end", "reply": "...", "confidence_score": 0.8, "escalate": false, "sources": [...]}\n\n"
    """
    # Prepare messages (same as chat method)
    messages = conversation_history.copy()
    messages.append({"role": "user", "content": message})

    lc_messages = []
    for msg in messages:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        else:
            lc_messages.append(AIMessage(content=msg["content"]))

    config = {
        "callbacks": [self.handler.handler],
        "metadata": {
            "langfuse_session_id": session_id or "default",
            "langfuse_tags": ["chat", "telco-agent", "stream"]
        }
    }

    full_reply = ""

    # Stream tokens from agent
    async for message_chunk, metadata in self.agent.astream({"messages": lc_messages}, config):
        if message_chunk.text:
            full_reply += message_chunk.text
            yield f"data: {json.dumps({'type': 'token', 'content': message_chunk.text})}\n\n"

    # Extract sources from the final result (need to get full result)
    # For now, we'll re-invoke to get the full result with tool outputs
    # This is not ideal but works for the simple case
    result = self.agent.invoke({"messages": lc_messages}, config)
    sources = self._extract_sources(result)

    # Determine escalate based on keywords in reply
    escalate_keywords = ["human agent", "speak to human", "representative", "escalate"]
    escalate = any(keyword.lower() in full_reply.lower() for keyword in escalate_keywords)

    # Heuristic confidence score
    confidence_score = 0.8 if sources else 0.5

    # Send final end event
    end_event = {
        "type": "end",
        "reply": full_reply,
        "confidence_score": confidence_score,
        "escalate": escalate,
        "sources": sources
    }
    yield f"data: {json.dumps(end_event)}\n\n"
```

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_chat.py::test_chat_stream_yields_sse_events -v`

Expected: PASS

**Step 5: Commit**

```bash
git add app/services/llm/chat.py tests/test_chat.py
git commit -m "feat: add ChatService.chat_stream() SSE generator"
```

---

## Task 3: Add SSE endpoint to routes

**Files:**
- Modify: `app/api/routes/chat.py`
- Test: `tests/test_api_routes.py` (create if not exists)

**Step 1: Write the failing test**

Create `tests/test_api_routes.py`:

```python
import pytest
import json
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.mark.asyncio
async def test_chat_stream_endpoint_sse_format():
    """POST /chat/stream should return SSE stream with token events"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = client.post(
            "/chat/stream",
            json={"message": "Hello", "conversation_history": [], "session_id": "test123"}
        )

        # Should return streaming response
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Parse SSE events
        events = []
        for line in response.iter_lines():
            if line.startswith(b"data: "):
                event_data = json.loads(line[6:])
                events.append(event_data)

        # Should have token events and end event
        assert len(events) > 0
        assert events[-1]["type"] == "end"
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_api_routes.py::test_chat_stream_endpoint_sse_format -v`

Expected: `404 Not Found` or similar (endpoint doesn't exist)

**Step 3: Write minimal implementation**

Add to `app/api/routes/chat.py`:

```python
from fastapi.responses import StreamingResponse

@router.post("/chat/stream")
async def stream_chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service)
):
    """Chat endpoint with SSE streaming for real-time token responses"""
    return StreamingResponse(
        service.chat_stream(
            message=request.message,
            conversation_history=request.conversation_history or [],
            session_id=request.session_id
        ),
        media_type="text/event-stream"
    )
```

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_api_routes.py::test_chat_stream_endpoint_sse_format -v`

Expected: PASS

**Step 5: Commit**

```bash
git add app/api/routes/chat.py tests/test_api_routes.py
git commit -m "feat: add /chat/stream SSE endpoint"
```

---

## Task 4: Add ChatService.chat_websocket() method

**Files:**
- Modify: `app/services/llm/chat.py`
- Test: `tests/test_chat.py`

**Step 1: Write the failing test**

Add to `tests/test_chat.py`:

```python
@pytest.mark.asyncio
async def test_chat_websocket_yields_json_events():
    """ChatService.chat_websocket() should handle websocket communication"""
    from unittest.mock import AsyncMock, MagicMock

    # Arrange
    vector_store = VectorStore()
    retriever_tool = RetrieverTool(vector_store=vector_store)
    agent = Agent(vector_store=vector_store, retriever_tool=retriever_tool)
    handler = CallbackHandler()
    service = ChatService(agent=agent, handler=handler)

    # Mock websocket
    websocket = MagicMock()
    websocket.receive_json = AsyncMock(return_value={
        "type": "message",
        "message": "Hello",
        "session_id": "test123",
        "conversation_history": []
    })
    websocket.send_json = AsyncMock()

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
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_chat.py::test_chat_websocket_yields_json_events -v`

Expected: `AttributeError: 'ChatService' object has no attribute 'chat_websocket'`

**Step 3: Write minimal implementation**

Add to `app/services/llm/chat.py` after `chat_stream()` method:

```python
async def chat_websocket(self, websocket, session_id: Optional[str] = None):
    """Handle WebSocket chat communication.

    Receives messages via websocket.receive_json():
    - {"type": "message", "message": "...", "conversation_history": [...]}
    - {"type": "cancel"}

    Sends responses via websocket.send_json():
    - {"type": "token", "content": "..."}
    - {"type": "end", "reply": "...", "confidence_score": 0.8, "escalate": false, "sources": [...]}
    - {"type": "error", "message": "..."}
    """
    try:
        # Receive initial message
        data = await websocket.receive_json()

        if data.get("type") == "cancel":
            await websocket.send_json({"type": "end", "reply": "", "cancelled": True})
            return

        if data.get("type") != "message":
            await websocket.send_json({"type": "error", "message": "Expected message type"})
            return

        message = data.get("message", "")
        conversation_history = data.get("conversation_history", [])
        session_id = data.get("session_id") or session_id

        # Prepare messages
        messages = conversation_history.copy()
        messages.append({"role": "user", "content": message})

        lc_messages = []
        for msg in messages:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            else:
                lc_messages.append(AIMessage(content=msg["content"]))

        config = {
            "callbacks": [self.handler.handler],
            "metadata": {
                "langfuse_session_id": session_id or "default",
                "langfuse_tags": ["chat", "telco-agent", "websocket"]
            }
        }

        full_reply = ""

        # Stream tokens
        async for message_chunk, _ in self.agent.astream({"messages": lc_messages}, config):
            # Check for cancel
            try:
                cancel_msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.01)
                if cancel_msg.get("type") == "cancel":
                    await websocket.send_json({"type": "end", "reply": full_reply, "cancelled": True})
                    return
            except asyncio.TimeoutError:
                pass

            if message_chunk.text:
                full_reply += message_chunk.text
                await websocket.send_json({"type": "token", "content": message_chunk.text})

        # Get final result for sources
        result = self.agent.invoke({"messages": lc_messages}, config)
        sources = self._extract_sources(result)

        # Determine escalate
        escalate_keywords = ["human agent", "speak to human", "representative", "escalate"]
        escalate = any(keyword.lower() in full_reply.lower() for keyword in escalate_keywords)
        confidence_score = 0.8 if sources else 0.5

        await websocket.send_json({
            "type": "end",
            "reply": full_reply,
            "confidence_score": confidence_score,
            "escalate": escalate,
            "sources": sources
        })

    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
```

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_chat.py::test_chat_websocket_yields_json_events -v`

Expected: PASS

**Step 5: Commit**

```bash
git add app/services/llm/chat.py tests/test_chat.py
git commit -m "feat: add ChatService.chat_websocket() method"
```

---

## Task 5: Add WebSocket endpoint to routes

**Files:**
- Modify: `app/api/routes/chat.py`
- Test: `tests/test_api_routes.py`

**Step 1: Write the failing test**

Add to `tests/test_api_routes.py`:

```python
@pytest.mark.asyncio
async def test_chat_websocket_endpoint():
    """WS /chat/stream/ws should handle websocket communication"""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    with client.websocket_connect("/chat/stream/ws") as websocket:
        # Send message
        websocket.send_json({
            "type": "message",
            "message": "Hello",
            "session_id": "test123",
            "conversation_history": []
        })

        # Receive events
        events = []
        while True:
            try:
                event = websocket.receive_json()
                events.append(event)
                if event.get("type") == "end":
                    break
            except:
                break

        # Should have received events
        assert len(events) > 0
        assert events[-1]["type"] == "end"
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_api_routes.py::test_chat_websocket_endpoint -v`

Expected: `404 Not Found` or similar (endpoint doesn't exist)

**Step 3: Write minimal implementation**

Add to `app/api/routes/chat.py`:

```python
from fastapi import WebSocket

@router.websocket("/chat/stream/ws")
async def stream_chat_websocket(
    websocket: WebSocket,
    service: ChatService = Depends(get_chat_service)
):
    """WebSocket endpoint for bidirectional chat streaming"""
    await websocket.accept()
    await service.chat_websocket(websocket)
```

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_api_routes.py::test_chat_websocket_endpoint -v`

Expected: PASS

**Step 5: Commit**

```bash
git add app/api/routes/chat.py tests/test_api_routes.py
git commit -m "feat: add /chat/stream/ws WebSocket endpoint"
```

---

## Task 6: Add error handling for streaming endpoints

**Files:**
- Modify: `app/services/llm/chat.py`
- Test: `tests/test_chat.py`

**Step 1: Write the failing test**

Add to `tests/test_chat.py`:

```python
@pytest.mark.asyncio
async def test_chat_stream_handles_errors():
    """ChatService.chat_stream() should handle errors gracefully"""
    # Arrange - mock agent that raises error
    from unittest.mock import AsyncMock, patch

    vector_store = VectorStore()
    retriever_tool = RetrieverTool(vector_store=vector_store)
    agent = Agent(vector_store=vector_store, retriever_tool=retriever_tool)
    handler = CallbackHandler()
    service = ChatService(agent=agent, handler=handler)

    # Mock astream to raise error
    async def mock_astream_error(*args, **kwargs):
        raise Exception("LLM error")

    with patch.object(agent, 'astream', side_effect=mock_astream_error):
        # Act
        events = []
        async for event_str in service.chat_stream("Hello", [], "test"):
            event_data = json.loads(event_str[6:])
            events.append(event_data)
            # Should receive error event and stop
            if event_data.get("type") == "error":
                break

        # Assert - should have error event
        assert len(events) > 0
        assert events[0]["type"] == "error"
        assert "LLM error" in events[0]["message"]
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_chat.py::test_chat_stream_handles_errors -v`

Expected: FAIL (error not caught, exception propagates)

**Step 3: Write minimal implementation**

Update `chat_stream()` method in `app/services/llm/chat.py` with error handling:

```python
async def chat_stream(self, message: str, conversation_history: list[dict], session_id: Optional[str] = None):
    """Stream chat response via Server-Sent Events."""
    try:
        # Prepare messages (same as chat method)
        messages = conversation_history.copy()
        messages.append({"role": "user", "content": message})

        lc_messages = []
        for msg in messages:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            else:
                lc_messages.append(AIMessage(content=msg["content"]))

        config = {
            "callbacks": [self.handler.handler],
            "metadata": {
                "langfuse_session_id": session_id or "default",
                "langfuse_tags": ["chat", "telco-agent", "stream"]
            }
        }

        full_reply = ""

        # Stream tokens from agent
        async for message_chunk, metadata in self.agent.astream({"messages": lc_messages}, config):
            if message_chunk.text:
                full_reply += message_chunk.text
                yield f"data: {json.dumps({'type': 'token', 'content': message_chunk.text})}\n\n"

        # Extract sources from the final result
        result = self.agent.invoke({"messages": lc_messages}, config)
        sources = self._extract_sources(result)

        # Determine escalate based on keywords
        escalate_keywords = ["human agent", "speak to human", "representative", "escalate"]
        escalate = any(keyword.lower() in full_reply.lower() for keyword in escalate_keywords)

        # Heuristic confidence score
        confidence_score = 0.8 if sources else 0.5

        # Send final end event
        end_event = {
            "type": "end",
            "reply": full_reply,
            "confidence_score": confidence_score,
            "escalate": escalate,
            "sources": sources
        }
        yield f"data: {json.dumps(end_event)}\n\n"

    except Exception as e:
        # Send error event
        error_event = {
            "type": "error",
            "message": str(e)
        }
        yield f"data: {json.dumps(error_event)}\n\n"
```

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_chat.py::test_chat_stream_handles_errors -v`

Expected: PASS

**Step 5: Commit**

```bash
git add app/services/llm/chat.py tests/test_chat.py
git commit -m "feat: add error handling for streaming"
```

---

## Task 7: Manual testing and verification

**Files:**
- None

**Step 1: Start the server**

Run: `poetry run python run.py`

Expected: Server starts on `http://0.0.0.0:8000`

**Step 2: Test SSE endpoint with curl**

Run:
```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I pay my bill?", "conversation_history": []}'
```

Expected: SSE stream with token events followed by end event

**Step 3: Test WebSocket endpoint with websocat**

If websocat is installed:
```bash
echo '{"type": "message", "message": "How do I pay my bill?", "conversation_history": []}' | websocat ws://localhost:8000/chat/stream/ws
```

Expected: JSON events streamed

**Step 4: Run all tests**

Run: `poetry run pytest tests/ -v`

Expected: All tests pass

**Step 5: Commit**

```bash
git add -A
git commit -m "test: add streaming implementation tests and manual verification"
```

---

## Summary

After completing all tasks:
- `Agent.astream()` - Async token streaming from LangChain
- `ChatService.chat_stream()` - SSE event generator
- `ChatService.chat_websocket()` - WebSocket handler
- `POST /chat/stream` - SSE endpoint
- `WS /chat/stream/ws` - WebSocket endpoint
- Error handling for both endpoints
- Full test coverage

**Total commits:** 7 (one per task)
