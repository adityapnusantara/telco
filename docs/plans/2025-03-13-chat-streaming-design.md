# Chat Streaming Design

**Date:** 2025-03-13
**Status:** Approved

## Overview

Add streaming endpoints to the Telco Customer Service AI Agent to provide real-time token-by-token responses, similar to ChatGPT's typing effect.

## Goals

- Stream response tokens as they are generated (token streaming)
- Use Server-Sent Events (SSE) for HTTP streaming
- Provide WebSocket alternative for bidirectional communication
- Maintain backward compatibility with existing `/chat` endpoint
- Send structured metadata (`confidence_score`, `escalate`, `sources`) at stream end

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI App                            │
├─────────────────────────────────────────────────────────────┤
│  POST /chat          (existing - unchanged)                 │
│  POST /chat/stream   (new - SSE)                            │
│  WS   /chat/stream/ws (new - WebSocket)                     │
├─────────────────────────────────────────────────────────────┤
│                    ChatService                              │
│  ├─ chat()           (existing - invoke)                    │
│  ├─ chat_stream()    (new - SSE generator)                  │
│  └─ chat_websocket() (new - WebSocket handler)              │
├─────────────────────────────────────────────────────────────┤
│                      Agent                                  │
│  ├─ invoke()         (existing)                             │
│  └─ astream()        (new - async stream method)            │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Agent.astream() Method

**File:** `app/services/llm/agent.py`

New async method that wraps LangChain's `astream()`:

```python
async def astream(self, messages, config):
    """Async stream tokens from the agent"""
    async for chunk in self._agent.astream(
        messages,
        config,
        stream_mode="messages",
        version="v2"
    ):
        yield chunk
```

### 2. ChatService.chat_stream() Method

**File:** `app/services/llm/chat.py`

SSE generator that yields token events:

```python
async def chat_stream(self, message: str, conversation_history: list, session_id: str = None):
    """Stream chat response via SSE"""
    # ... setup messages and config ...

    full_reply = ""
    async for message_chunk, metadata in self.agent.astream({"messages": lc_messages}, config):
        if message_chunk.text:
            full_reply += message_chunk.text
            yield f"data: {json.dumps({'type': 'token', 'content': message_chunk.text})}\n\n"

    # Send final metadata
    sources = []  # extract from tool results
    yield f"data: {json.dumps({'type': 'end', 'reply': full_reply, 'confidence_score': 0.8, 'escalate': False, 'sources': sources})}\n\n"
```

### 3. ChatService.chat_websocket() Method

WebSocket handler for bidirectional communication.

### 4. SSE Endpoint

**File:** `app/api/routes/chat.py`

```python
from fastapi.responses import StreamingResponse

@router.post("/chat/stream")
async def stream_chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service)
):
    return StreamingResponse(
        service.chat_stream(request.message, request.conversation_history, request.session_id),
        media_type="text/event-stream"
    )
```

### 5. WebSocket Endpoint

```python
from fastapi import WebSocket

@router.websocket("/chat/stream/ws")
async def stream_chat_websocket(websocket: WebSocket, service: ChatService = Depends(get_chat_service)):
    await websocket.accept()
    # ... handle WebSocket communication ...
```

## Event Format

### SSE (`/chat/stream`)

**Request:**
```json
POST /chat/stream
{
  "message": "How do I pay my bill?",
  "session_id": "abc123",
  "conversation_history": []
}
```

**Response (stream):**
```
data: {"type": "token", "content": "To"}
data: {"type": "token", "content": " pay"}
data: {"type": "token", "content": " your"}
...
data: {"type": "end", "reply": "To pay your bill...", "confidence_score": 0.95, "escalate": false, "sources": ["billing_qna.json"]}
```

### WebSocket (`/chat/stream/ws`)

**Client → Server:**
```json
{"type": "message", "message": "How do I pay my bill?", "session_id": "abc123"}
```

**Server → Client:**
```json
{"type": "token", "content": "To"}
{"type": "token", "content": " pay"}
...
{"type": "end", "reply": "...", "confidence_score": 0.95, "escalate": false, "sources": [...]}
```

**Additional WebSocket messages:**
- `{"type": "cancel"}` - Cancel mid-stream
- `{"type": "ping"}` - Keep-alive

## Error Handling

| Scenario | SSE Behavior | WebSocket Behavior |
|----------|--------------|-------------------|
| LLM error mid-stream | `{"type": "error", "message": "..."}` + close | Same + close connection |
| Retriever fails | Log to Langfuse, send error event | Same |
| Client disconnect | Catch `GeneratorExit`, clean up | Handle `WebSocketDisconnect` |
| Timeout (30s) | `{"type": "error", "message": "Request timeout"}` | Same |
| Invalid request | HTTP 422 immediately | Close with error frame |

## Metadata Strategy

Since we're streaming tokens (not using `response_format`), metadata fields will be populated as follows:

- **`reply`**: Accumulated full text from all tokens
- **`confidence_score`**: 0.8 if sources found, 0.5 if no sources (heuristic, can be refined)
- **`escalate`**: Pattern matching on final reply (keyword detection for escalation)
- **`sources`**: Extracted from tool results as in existing implementation

## Testing

### Unit Tests (`tests/test_streaming.py`)

1. `test_agent_astream_yields_tokens()` - Verify Agent.astream() works
2. `test_chat_stream_sse_format()` - Verify SSE event format
3. `test_chat_stream_endpoint()` - Test SSE endpoint with httpx
4. `test_chat_websocket_endpoint()` - Test WebSocket endpoint

### Manual Testing

```bash
# SSE test
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I pay my bill?"}'

# WebSocket test
websocat ws://localhost:8000/chat/stream/ws
```

## Implementation Files

| File | Changes |
|------|---------|
| `app/services/llm/agent.py` | Add `astream()` method |
| `app/services/llm/chat.py` | Add `chat_stream()`, `chat_websocket()` methods |
| `app/api/routes/chat.py` | Add SSE and WebSocket endpoints |
| `tests/test_streaming.py` | New test file |
| `docs/plans/2025-03-13-chat-streaming-implementation.md` | Implementation plan |

## Backward Compatibility

- Existing `/chat` endpoint remains unchanged
- Existing response models (`ChatResponse`) unchanged
- No breaking changes to existing API
