# Refactor Global Singletons to Class-Based Services

**Date:** 2025-03-11
**Status:** Approved

## Overview

Refactor existing singleton pattern using module-level global variables to class-based services with explicit dependency injection and lifecycle management using FastAPI's `app.state`.

## Motivation

- **Cleaner dependency management** - Dependencies are explicit via constructor injection
- **Better state management** - Explicit lifecycle control through FastAPI startup/shutdown events
- **No global variables** - All service instances stored in `app.state`

## Current State

Existing singleton pattern with global variables:

```python
# app/services/llm/agent.py
_agent = None

def get_agent():
    global _agent
    if _agent is None:
        _agent = create_agent(...)
    return _agent
```

Similar pattern in:
- `app/services/llm/agent.py` - `_agent` global
- `app/services/llm/callbacks.py` - `_handler` global
- `app/services/rag/vector_store.py` - `_qdrant_client` and `_vector_store` global

## Target Design

### Architecture

```
FastAPI startup event
    ↓
app.state.vector_store = VectorStore(...)
app.state.agent = Agent(vector_store=app.state.vector_store)
app.state.chat_service = ChatService(agent=app.state.agent, handler=...)
    ↓
Routes use Depends(get_chat_service) → request.app.state.chat_service
```

### Component Classes

#### VectorStore Class

```python
# app/services/rag/vector_store.py
class VectorStore:
    def __init__(self, qdrant_url: str, qdrant_api_key: str, collection_name: str):
        self._client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)
        self._store = QdrantVectorStore(
            client=self._client,
            collection_name=collection_name,
            embedding=self._embeddings
        )

    @property
    def store(self):
        return self._store
```

#### CallbackHandler Class

```python
# app/services/llm/callbacks.py
class CallbackHandler:
    def __init__(self):
        self._handler = LangfuseCallbackHandler()

    @property
    def handler(self):
        return self._handler
```

#### Agent Class

```python
# app/services/llm/agent.py
class Agent:
    def __init__(self, vector_store: VectorStore):
        self._vector_store = vector_store
        self._llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self._agent = create_agent(
            model=self._llm,
            tools=[self._get_retriever_tool()],
            system_prompt=self._get_system_prompt()
        )

    def invoke(self, messages, config):
        return self._agent.invoke(messages, config)
```

#### ChatService Class

```python
# app/services/llm/chat.py
class ChatService:
    def __init__(self, agent: Agent, handler: CallbackHandler):
        self.agent = agent
        self.handler = handler

    def chat(self, message: str, conversation_history: list, session_id: str = None):
        messages = conversation_history.copy()
        messages.append({"role": "user", "content": message})

        config = {
            "callbacks": [self.handler.handler],
            "metadata": {
                "langfuse_session_id": session_id or "default",
                "langfuse_tags": ["chat", "telco-agent"]
            }
        }

        result = self.agent.invoke({"messages": messages}, config=config)
        reply = result.messages[-1]["content"]
        escalate = self._should_escalate(reply)

        return ChatResponse(reply=reply, escalate=escalate)

    def _should_escalate(self, reply: str) -> bool:
        escalation_indicators = [
            "cannot help", "don't know", "unable to assist",
            "speak to a human", "transfer to agent", "escalate"
        ]
        reply_lower = reply.lower()
        return any(indicator in reply_lower for indicator in escalation_indicators)
```

### Initialization in main.py

```python
# app/main.py
from fastapi import FastAPI
from app.core.config import config
from app.services.rag.vector_store import VectorStore
from app.services.llm.callbacks import CallbackHandler
from app.services.llm.agent import Agent
from app.services.llm.chat import ChatService

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    try:
        app.state.vector_store = VectorStore(
            qdrant_url=config.QDRANT_URL,
            qdrant_api_key=config.QDRANT_API_KEY,
            collection_name=config.QDRANT_COLLECTION_NAME
        )
        logger.info("VectorStore initialized")

        app.state.callback_handler = CallbackHandler()
        logger.info("CallbackHandler initialized")

        app.state.agent = Agent(vector_store=app.state.vector_store)
        logger.info("Agent initialized")

        app.state.chat_service = ChatService(
            agent=app.state.agent,
            handler=app.state.callback_handler
        )
        logger.info("ChatService initialized")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    if hasattr(app.state, 'vector_store'):
        # Cleanup if needed
        pass
```

### Route Updates

```python
# app/api/routes/chat.py
from fastapi import APIRouter, Depends, Request
from app.api.models import ChatRequest, ChatResponse

router = APIRouter()

def get_chat_service(request: Request) -> ChatService:
    service = request.app.state.chat_service
    if service is None:
        raise HTTPException(status_code=503, detail="Services not initialized")
    return service

@router.post("/chat", response_model=ChatResponse)
async def create_chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service)
) -> ChatResponse:
    return service.chat(
        message=request.message,
        conversation_history=request.conversation_history or [],
        session_id=request.session_id
    )
```

## Error Handling

1. **Initialization failures** - Logged and re-raised to prevent FastAPI from starting with broken services
2. **Access before initialization** - Checked in `get_chat_service()` dependency, returns 503 if not initialized
3. **Graceful shutdown** - Cleanup via `shutdown` event if needed

## Testing

### Unit Tests

```python
# tests/test_chat_service.py
from unittest.mock import Mock
from app.services.llm.chat import ChatService

def test_chat_service():
    mock_agent = Mock()
    mock_agent.invoke.return_value = Mock(messages=[{"content": "test reply"}])
    mock_handler = Mock()

    service = ChatService(agent=mock_agent, handler=mock_handler)
    result = service.chat("hello", [], "conv-123")

    assert result.reply == "test reply"
```

### Integration Tests

```python
# tests/test_chat_endpoint.py
from fastapi.testclient import TestClient
from app.main import app

def test_chat_endpoint():
    from unittest.mock import Mock

    mock_service = Mock()
    mock_service.chat.return_value = ChatResponse(reply="test", escalate=False)

    app.state.chat_service = mock_service

    client = TestClient(app)
    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 200
    assert response.json()["reply"] == "test"
```

## Files to Modify

1. `app/services/rag/vector_store.py` - Convert to class
2. `app/services/llm/callbacks.py` - Convert to class
3. `app/services/llm/agent.py` - Convert to class
4. `app/services/llm/chat.py` - Convert to class with DI
5. `app/main.py` - Add startup/shutdown events with `app.state`
6. `app/api/routes/chat.py` - Use `Depends(get_chat_service)`

## Files to Update Tests

1. `tests/test_vector_store.py` - Update for VectorStore class
2. `tests/test_callbacks.py` - Update for CallbackHandler class
3. `tests/test_agent.py` - Update for Agent class
4. `tests/test_chat_service.py` - Update for ChatService class
5. `tests/test_chat_endpoint.py` - Update for app.state approach
6. `tests/conftest.py` - Update fixtures for new pattern
