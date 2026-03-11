# Refactor Global Singletons to Class-Based Services - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor singleton pattern using module-level global variables to class-based services with explicit dependency injection and lifecycle management using FastAPI's `app.state`.

**Architecture:** Convert `get_*()` functions to classes with `__init__()` accepting dependencies via constructor. Store service instances in FastAPI's `app.state` during startup event. Routes use `Depends()` to inject services.

**Tech Stack:** FastAPI, LangChain v1.2.11, Qdrant Cloud, Langfuse, Pytest

---

### Task 1: Create VectorStore Class

**Files:**
- Modify: `app/services/rag/vector_store.py`

**Step 1: Write the failing test**

```python
# tests/test_vector_store.py
import pytest
from app.services.rag.vector_store import VectorStore
from app.core.config import config

def test_vector_store_init():
    """Test VectorStore class initialization"""
    store = VectorStore(
        qdrant_url=config.QDRANT_URL,
        qdrant_api_key=config.QDRANT_API_KEY,
        collection_name=config.QDRANT_COLLECTION_NAME
    )

    assert store.store is not None
    assert hasattr(store, '_client')
    assert hasattr(store, '_embeddings')
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_vector_store.py::test_vector_store_init -v`
Expected: FAIL with "VectorStore not defined" or AttributeError

**Step 3: Write minimal implementation**

```python
# app/services/rag/vector_store.py
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient


class VectorStore:
    """Qdrant vector store wrapper with explicit initialization"""

    def __init__(self, qdrant_url: str, qdrant_api_key: str, collection_name: str):
        self._client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        self._embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=512,
        )
        self._store = QdrantVectorStore(
            client=self._client,
            collection_name=collection_name,
            embedding=self._embeddings,
        )

    @property
    def store(self):
        """Get the underlying LangChain vector store"""
        return self._store


# Legacy function for backward compatibility (will be removed after migration)
def get_qdrant_client():
    """Deprecated: Use VectorStore class instead"""
    raise DeprecationWarning("Use VectorStore class instead")


def get_vector_store():
    """Deprecated: Use VectorStore class instead"""
    raise DeprecationWarning("Use VectorStore class instead")
```

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_vector_store.py::test_vector_store_init -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/services/rag/vector_store.py tests/test_vector_store.py
git commit -m "feat: create VectorStore class

Convert from singleton function to class-based approach with
explicit constructor injection. Deprecate old get_*() functions.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Create CallbackHandler Class

**Files:**
- Modify: `app/services/llm/callbacks.py`

**Step 1: Write the failing test**

```python
# tests/test_callbacks.py
import pytest
from app.services.llm.callbacks import CallbackHandler

def test_callback_handler_init():
    """Test CallbackHandler class initialization"""
    handler = CallbackHandler()

    assert handler.handler is not None
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_callbacks.py::test_callback_handler_init -v`
Expected: FAIL with "CallbackHandler class not defined"

**Step 3: Write minimal implementation**

```python
# app/services/llm/callbacks.py
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler


class CallbackHandler:
    """Langfuse callback handler wrapper with explicit initialization"""

    def __init__(self):
        self._handler = LangfuseCallbackHandler()

    @property
    def handler(self):
        """Get the underlying Langfuse callback handler"""
        return self._handler


# Legacy function for backward compatibility (will be removed after migration)
def get_langfuse_handler():
    """Deprecated: Use CallbackHandler class instead"""
    raise DeprecationWarning("Use CallbackHandler class instead")
```

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_callbacks.py::test_callback_handler_init -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/services/llm/callbacks.py tests/test_callbacks.py
git commit -m "feat: create CallbackHandler class

Convert from singleton function to class-based approach with
explicit initialization. Deprecate old get_*() function.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Create Agent Class

**Files:**
- Modify: `app/services/llm/agent.py`

**Step 1: Write the failing test**

```python
# tests/test_agent.py
import pytest
from unittest.mock import Mock, patch
from app.services.llm.agent import Agent
from app.services.rag.vector_store import VectorStore

def test_agent_init():
    """Test Agent class initialization"""
    mock_vector_store = Mock(spec=VectorStore)

    with patch('app.services.llm.agent.get_system_prompt') as mock_prompt, \
         patch('app.services.llm.agent.get_retriever_tool') as mock_tool:
        mock_prompt.return_value = [{"content": "You are a helpful assistant"}]
        mock_tool.return_value = Mock()

        agent = Agent(vector_store=mock_vector_store)

        assert agent._vector_store == mock_vector_store
        assert agent._agent is not None
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_agent.py::test_agent_init -v`
Expected: FAIL with "Agent class not defined"

**Step 3: Write minimal implementation**

```python
# app/services/llm/agent.py
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from app.prompts.langfuse import get_system_prompt
from app.services.rag.retriever import get_retriever_tool
from app.services.rag.vector_store import VectorStore


class Agent:
    """LangChain agent wrapper with explicit initialization and DI"""

    def __init__(self, vector_store: VectorStore):
        self._vector_store = vector_store
        self._llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self._system_prompt = get_system_prompt()

        # Build prompt text
        prompt_text = "\n".join([msg["content"] for msg in self._system_prompt])

        # Get retriever tool
        self._retriever_tool = get_retriever_tool()

        # Create the agent
        self._agent = create_agent(
            model=self._llm,
            tools=[self._retriever_tool],
            system_prompt=prompt_text,
        )

    def invoke(self, messages, config):
        """Invoke the agent with messages and config"""
        return self._agent.invoke(messages, config)


# Legacy function for backward compatibility (will be removed after migration)
def get_agent():
    """Deprecated: Use Agent class instead"""
    raise DeprecationWarning("Use Agent class instead")
```

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_agent.py::test_agent_init -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/services/llm/agent.py tests/test_agent.py
git commit -m "feat: create Agent class

Convert from singleton function to class-based approach with
constructor dependency injection. Deprecate old get_*() function.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Create ChatService Class

**Files:**
- Modify: `app/services/llm/chat.py`

**Step 1: Write the failing test**

```python
# tests/test_chat_service.py
import pytest
from unittest.mock import Mock
from app.services.llm.chat import ChatService
from app.services.llm.agent import Agent
from app.services.llm.callbacks import CallbackHandler

def test_chat_service_init():
    """Test ChatService class initialization"""
    mock_agent = Mock(spec=Agent)
    mock_handler = Mock(spec=CallbackHandler)

    service = ChatService(agent=mock_agent, handler=mock_handler)

    assert service.agent == mock_agent
    assert service.handler == mock_handler

def test_chat_service_chat():
    """Test ChatService.chat method"""
    mock_agent = Mock()
    mock_agent.invoke.return_value = Mock(messages=[{"content": "Hello! How can I help?"}])
    mock_handler = Mock()
    mock_handler.handler = Mock()

    service = ChatService(agent=mock_agent, handler=mock_handler)
    result = service.chat("Hello", [], "conv-123")

    assert result.reply == "Hello! How can I help?"
    assert result.escalate is False
    mock_agent.invoke.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_chat_service.py -v`
Expected: FAIL with "ChatService class not defined"

**Step 3: Write minimal implementation**

```python
# app/services/llm/chat.py
from pydantic import BaseModel
from typing import Optional
from app.services.llm.agent import Agent
from app.services.llm.callbacks import CallbackHandler


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    reply: str
    escalate: bool
    sources: Optional[list[str]] = None
    confidence_score: Optional[float] = None


class ChatService:
    """Chat service with explicit dependency injection"""

    def __init__(self, agent: Agent, handler: CallbackHandler):
        self.agent = agent
        self.handler = handler

    def chat(self, message: str, conversation_history: list[dict], conversation_id: Optional[str] = None) -> ChatResponse:
        """Process a chat message using the RAG agent"""
        messages = conversation_history.copy()
        messages.append({"role": "user", "content": message})

        config = {
            "callbacks": [self.handler.handler],
            "metadata": {
                "langfuse_session_id": conversation_id or "default",
                "langfuse_tags": ["chat", "telco-agent"]
            }
        }

        result = self.agent.invoke({"messages": messages}, config=config)
        reply = result.messages[-1]["content"]

        escalate = self._should_escalate(reply)

        return ChatResponse(reply=reply, escalate=escalate)

    def _should_escalate(self, reply: str) -> bool:
        """Determine if escalation is needed"""
        escalation_indicators = [
            "cannot help", "don't know", "unable to assist",
            "speak to a human", "transfer to agent", "escalate"
        ]
        reply_lower = reply.lower()
        return any(indicator in reply_lower for indicator in escalation_indicators)
```

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_chat_service.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/services/llm/chat.py tests/test_chat_service.py
git commit -m "feat: create ChatService class

Convert from function to class-based approach with constructor
dependency injection. Move ChatResponse into same file.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 5: Update main.py with Startup/Shutdown Events

**Files:**
- Modify: `app/main.py`

**Step 1: Write the failing test**

```python
# tests/test_main.py
import pytest
from app.main import app
from app.services.rag.vector_store import VectorStore
from app.services.llm.callbacks import CallbackHandler
from app.services.llm.agent import Agent
from app.services.llm.chat import ChatService

def test_startup_creates_services():
    """Test that startup event creates all services in app.state"""
    from fastapi.testclient import TestClient

    # Trigger startup
    with TestClient(app) as client:
        # Check services are initialized
        assert hasattr(app.state, 'vector_store')
        assert hasattr(app.state, 'callback_handler')
        assert hasattr(app.state, 'agent')
        assert hasattr(app.state, 'chat_service')

        assert isinstance(app.state.vector_store, VectorStore)
        assert isinstance(app.state.callback_handler, CallbackHandler)
        assert isinstance(app.state.agent, Agent)
        assert isinstance(app.state.chat_service, ChatService)
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_main.py::test_startup_creates_services -v`
Expected: FAIL with app.state missing services

**Step 3: Write minimal implementation**

```python
# app/main.py
import logging
from fastapi import FastAPI
from app.core.config import config
from app.services.rag.vector_store import VectorStore
from app.services.llm.callbacks import CallbackHandler
from app.services.llm.agent import Agent
from app.services.llm.chat import ChatService

logger = logging.getLogger(__name__)

app = FastAPI(title="Telco Customer Service AI Agent")


@app.on_event("startup")
async def startup_event():
    """Initialize all service instances and store in app.state"""
    try:
        logger.info("Initializing VectorStore...")
        app.state.vector_store = VectorStore(
            qdrant_url=config.QDRANT_URL,
            qdrant_api_key=config.QDRANT_API_KEY,
            collection_name=config.QDRANT_COLLECTION_NAME
        )

        logger.info("Initializing CallbackHandler...")
        app.state.callback_handler = CallbackHandler()

        logger.info("Initializing Agent...")
        app.state.agent = Agent(vector_store=app.state.vector_store)

        logger.info("Initializing ChatService...")
        app.state.chat_service = ChatService(
            agent=app.state.agent,
            handler=app.state.callback_handler
        )

        logger.info("All services initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup service connections if needed"""
    logger.info("Shutting down services...")
    # Add cleanup logic here if needed in the future
    logger.info("Shutdown complete")


@app.get("/")
async def root():
    """Root endpoint with app info"""
    return {
        "app": "Telco Customer Service AI Agent",
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}
```

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_main.py::test_startup_creates_services -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/main.py tests/test_main.py
git commit -m "feat: add startup/shutdown events with app.state

Initialize all services in FastAPI startup event and store
in app.state. No more global variables. Add logging for
service initialization.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 6: Update Chat Route to Use Depends

**Files:**
- Modify: `app/api/routes/chat.py`

**Step 1: Write the failing test**

```python
# tests/test_chat_endpoint.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

def test_chat_endpoint_with_depends():
    """Test chat endpoint using Depends injection"""
    from unittest.mock import Mock
    from app.services.llm.chat import ChatResponse

    # Setup mock service
    mock_service = Mock()
    mock_service.chat.return_value = ChatResponse(
        reply="Test response",
        escalate=False
    )
    app.state.chat_service = mock_service

    with TestClient(app) as client:
        response = client.post("/chat", json={
            "message": "Hello",
            "conversation_history": [],
            "conversation_id": "test-123"
        })

        assert response.status_code == 200
        assert response.json()["reply"] == "Test response"
        assert response.json()["escalate"] is False

def test_chat_endpoint_returns_503_if_not_initialized():
    """Test that chat endpoint returns 503 if services not initialized"""
    app.state.chat_service = None

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "Hello"})

        assert response.status_code == 503
        assert "not initialized" in response.json()["detail"]
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_chat_endpoint.py -v`
Expected: FAIL with current implementation not using Depends

**Step 3: Write minimal implementation**

```python
# app/api/routes/chat.py
from fastapi import APIRouter, HTTPException, Depends, Request
from app.api.models import ChatRequest, ChatResponse
from app.services.llm.chat import ChatService

router = APIRouter()


def get_chat_service(request: Request) -> ChatService:
    """Dependency to get ChatService from app.state"""
    service = request.app.state.chat_service
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Services not initialized"
        )
    return service


@router.post("/chat", response_model=ChatResponse)
async def create_chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service)
) -> ChatResponse:
    """Chat endpoint for the Telco customer service agent"""
    try:
        response = service.chat(
            message=request.message,
            conversation_history=request.conversation_history or [],
            conversation_id=request.conversation_id
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_chat_endpoint.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/api/routes/chat.py tests/test_chat_endpoint.py
git commit -m "feat: update chat route to use Depends injection

Use FastAPI Depends to inject ChatService from app.state.
Add 503 error handling for uninitialized services.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 7: Update Retriever to Use VectorStore Class

**Files:**
- Modify: `app/services/rag/retriever.py`

**Step 1: Write the failing test**

```python
# tests/test_retriever.py
import pytest
from unittest.mock import Mock
from app.services.rag.retriever import RetrieverTool
from app.services.rag.vector_store import VectorStore

def test_retriever_tool_init():
    """Test RetrieverTool class initialization"""
    mock_store = Mock(spec=VectorStore)
    mock_store.store = Mock()

    tool = RetrieverTool(vector_store=mock_store)

    assert tool._vector_store == mock_store
    assert tool.tool.name == "search_knowledge_base"
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_retriever.py -v`
Expected: FAIL - need to check current retriever implementation

**Step 3: Update retriever implementation**

First check current implementation:
```bash
cat app/services/rag/retriever.py
```

Then update based on current code (assuming it uses get_vector_store()):
```python
# app/services/rag/retriever.py
from langchain_core.tools import tool
from app.services.rag.vector_store import VectorStore


class RetrieverTool:
    """Wrapper for LangChain retriever tool with DI"""

    def __init__(self, vector_store: VectorStore):
        self._vector_store = vector_store
        self._tool = self._create_tool()

    def _create_tool(self):
        """Create the LangChain tool"""
        @tool
        def search_knowledge_base(query: str) -> str:
            """Search the knowledge base for answers to customer questions.

            Args:
                query: The customer's question or search query

            Returns:
                Relevant information from the knowledge base
            """
            retriever = self._vector_store.store.as_retriever(
                search_kwargs={"k": 3}
            )
            docs = retriever.invoke(query)
            return "\n\n".join([doc.page_content for doc in docs])

        return search_knowledge_base

    @property
    def tool(self):
        """Get the LangChain tool"""
        return self._tool


# Legacy function for backward compatibility
def get_retriever_tool():
    """Get the retriever tool (legacy, uses singleton)"""
    from app.services.rag.vector_store import get_vector_store
    vector_store = get_vector_store()
    retriever = RetrieverTool(vector_store)
    return retriever.tool
```

**Step 4: Update Agent class to use RetrieverTool**

```python
# app/services/llm/agent.py - add RetrieverTool DI
class Agent:
    def __init__(self, vector_store: VectorStore, retriever_tool=None):
        self._vector_store = vector_store
        self._llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self._system_prompt = get_system_prompt()

        # Build prompt text
        prompt_text = "\n".join([msg["content"] for msg in self._system_prompt])

        # Get or create retriever tool
        if retriever_tool:
            self._retriever_tool = retriever_tool
        else:
            from app.services.rag.retriever import RetrieverTool
            self._retriever_tool = RetrieverTool(vector_store).tool

        # Create the agent
        self._agent = create_agent(
            model=self._llm,
            tools=[self._retriever_tool],
            system_prompt=prompt_text,
        )
```

**Step 5: Run test to verify it passes**

Run: `poetry run pytest tests/test_retriever.py -v`
Expected: PASS

**Step 6: Update main.py to pass RetrieverTool**

```python
# app/main.py - update startup
from app.services.rag.retriever import RetrieverTool

@app.on_event("startup")
async def startup_event():
    try:
        # ... existing VectorStore init ...

        logger.info("Initializing RetrieverTool...")
        retriever_tool = RetrieverTool(vector_store=app.state.vector_store)

        logger.info("Initializing Agent...")
        app.state.agent = Agent(
            vector_store=app.state.vector_store,
            retriever_tool=retriever_tool.tool
        )
        # ...
```

**Step 7: Commit**

```bash
git add app/services/rag/retriever.py app/services/llm/agent.py app/main.py tests/test_retriever.py
git commit -m "feat: add RetrieverTool class with DI

Create RetrieverTool class accepting VectorStore via
constructor. Update Agent to accept optional RetrieverTool.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 8: Run Full Test Suite

**Files:**
- None (verification only)

**Step 1: Run all tests**

```bash
poetry run pytest -v
```

**Expected:** All tests pass

**Step 2: Run tests with coverage**

```bash
poetry run pytest --cov=app --cov-report=html
```

**Expected:** Coverage report shows all new classes are tested

**Step 3: Verify application starts**

```bash
poetry run python run.py
```

**Expected:** Application starts without errors, logs show all services initialized

**Step 4: Test chat endpoint manually**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are your plans?"}'
```

**Expected:** Successful response

**Step 5: Commit if any fixes needed**

```bash
# Only if fixes were needed
git add .
git commit -m "fix: address issues found in integration testing

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 9: Remove Legacy Functions

**Files:**
- `app/services/rag/vector_store.py`
- `app/services/llm/callbacks.py`
- `app/services/llm/agent.py`

**Step 1: Verify no code uses legacy functions**

```bash
grep -r "get_vector_store\|get_qdrant_client\|get_langfuse_handler\|get_agent" app/ tests/ --exclude-dir=".git"
```

**Expected:** No results (or only in the files we're about to modify)

**Step 2: Remove legacy functions**

```python
# app/services/rag/vector_store.py - remove these lines:
# def get_qdrant_client(): ...
# def get_vector_store(): ...

# app/services/llm/callbacks.py - remove this line:
# def get_langfuse_handler(): ...

# app/services/llm/agent.py - remove this line:
# def get_agent(): ...
```

**Step 3: Run tests to verify**

```bash
poetry run pytest -v
```

**Expected:** All tests still pass

**Step 4: Commit**

```bash
git add app/services/rag/vector_store.py app/services/llm/callbacks.py app/services/llm/agent.py
git commit -m "refactor: remove legacy singleton functions

Remove deprecated get_*() functions. All code now uses
class-based approach with DI.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 10: Update Documentation

**Files:**
- `README.md`
- `CLAUDE.md`

**Step 1: Update README.md**

Add new architecture documentation:
```markdown
## Architecture

The application uses class-based services with dependency injection:

- **VectorStore** - Qdrant vector store wrapper
- **CallbackHandler** - Langfuse tracing handler
- **Agent** - LangChain agent with retriever tool
- **ChatService** - Chat orchestration with escalation detection

All services are initialized during FastAPI startup and stored in `app.state`.
Routes use FastAPI's `Depends()` for dependency injection.
```

**Step 2: Update CLAUDE.md**

Update architecture section to reflect new class-based approach

**Step 3: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: update architecture documentation

Reflect new class-based service architecture with DI and
app.state lifecycle management.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Verification Checklist

After completing all tasks:

- [ ] All new classes have tests
- [ ] All tests pass (`poetry run pytest`)
- [ ] No global variables in service files
- [ ] Services stored in `app.state`
- [ ] Routes use `Depends()` for injection
- [ ] Application starts without errors
- [ ] Chat endpoint works end-to-end
- [ ] Documentation updated

---

**Plan complete and saved to `docs/plans/2025-03-11-refactor-singletons-to-classes-implementation.md`.**

**Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
