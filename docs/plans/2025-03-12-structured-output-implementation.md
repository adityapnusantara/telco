# Structured Output Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `response_format` to LangChain's `create_agent` to return structured responses with `reply`, `confidence_score`, and `escalate` fields directly from GPT-4o, eliminating post-processing and keyword-based escalation detection.

**Architecture:** Pass a Pydantic model (`StructuredChatResponse`) to `create_agent`'s `response_format` parameter. GPT-4o's native structured output will return validated data in `result["structured_response"]`. `ChatService` extracts this structured response and sources from tool calls.

**Tech Stack:** LangChain v1.2.11, Pydantic, OpenAI GPT-4o, FastAPI, pytest

---

## Task 1: Add StructuredChatResponse Model to Agent

**Files:**
- Modify: `app/services/llm/agent.py`

**Step 1: Add the StructuredChatResponse Pydantic model**

Add the model definition at the top of `app/services/llm/agent.py` (after imports, before `Agent` class):

```python
class StructuredChatResponse(BaseModel):
    """Structured response schema for agent output"""
    reply: str = Field(description="The natural language response to the user's question")
    confidence_score: float = Field(
        description="Confidence score from 0.0 to 1.0 indicating how certain the agent is about its answer",
        ge=0.0,
        le=1.0
    )
    escalate: bool = Field(description="True if the user should be escalated to a human agent")
```

**Step 2: Add Field import to imports**

Update the imports at the top of the file to include `Field` from pydantic:

```python
from pydantic import BaseModel, Field
```

**Step 3: Update imports to include pydantic**

The file should now start with:

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from app.prompts.langfuse import get_system_prompt
from app.services.rag.retriever import RetrieverTool
from app.services.rag.vector_store import VectorStore
```

**Step 4: Commit**

```bash
git add app/services/llm/agent.py
git commit -m "feat: add StructuredChatResponse model for structured output"
```

---

## Task 2: Pass response_format to create_agent

**Files:**
- Modify: `app/services/llm/agent.py:100-105`

**Step 1: Update the create_agent call to include response_format**

Modify the `create_agent` call in `Agent.__init__` to include `response_format`:

```python
self._agent = create_agent(
    model=self._llm,
    tools=[self._retriever_tool],
    system_prompt=self._system_prompt,
    response_format=StructuredChatResponse
)
```

The `__init__` method should now look like:

```python
def __init__(self, vector_store: VectorStore, retriever_tool=None):
    self._vector_store = vector_store
    self._llm = ChatOpenAI(model="gpt-4o", temperature=0)
    self._system_prompt = get_system_prompt()

    if retriever_tool:
        self._retriever_tool = retriever_tool.tool
    else:
        # Create RetrieverTool if not provided
        self._retriever_tool = RetrieverTool(vector_store).tool

    self._agent = create_agent(
        model=self._llm,
        tools=[self._retriever_tool],
        system_prompt=self._system_prompt,
        response_format=StructuredChatResponse
    )
```

**Step 2: Commit**

```bash
git add app/services/llm/agent.py
git commit -m "feat: pass response_format to create_agent for structured output"
```

---

## Task 3: Write Failing Test for Structured Response

**Files:**
- Create: `tests/test_agent_structured_output.py`

**Step 1: Write the failing test**

Create `tests/test_agent_structured_output.py`:

```python
import pytest
from app.services.llm.agent import Agent, StructuredChatResponse
from app.services.rag.vector_store import VectorStore


def test_structured_chat_response_model_exists():
    """Test that StructuredChatResponse model is defined"""
    from app.services.llm.agent import StructuredChatResponse
    assert hasattr(StructuredChatResponse, 'model_fields')
    assert 'reply' in StructuredChatResponse.model_fields
    assert 'confidence_score' in StructuredChatResponse.model_fields
    assert 'escalate' in StructuredChatResponse.model_fields


def test_structured_chat_response_confidence_validation():
    """Test that confidence_score must be between 0 and 1"""
    from app.services.llm.agent import StructuredChatResponse

    # Valid range
    valid = StructuredChatResponse(
        reply="Test response",
        confidence_score=0.8,
        escalate=False
    )
    assert valid.confidence_score == 0.8

    # Invalid: too high
    with pytest.raises(Exception):
        StructuredChatResponse(
            reply="Test response",
            confidence_score=1.5,
            escalate=False
        )

    # Invalid: too low
    with pytest.raises(Exception):
        StructuredChatResponse(
            reply="Test response",
            confidence_score=-0.1,
            escalate=False
        )


def test_agent_has_response_format_configured(monkeypatch):
    """Test that Agent is initialized with response_format"""
    # This test will fail initially, then pass after we update the code
    from unittest.mock import MagicMock, patch

    # Mock the dependencies
    mock_vector_store = MagicMock(spec=VectorStore)
    mock_llm = MagicMock()
    mock_system_prompt = "Test prompt"

    with patch('app.services.llm.agent.ChatOpenAI') as mock_chat_openai, \
         patch('app.services.llm.agent.get_system_prompt') as mock_get_prompt, \
         patch('app.services.llm.agent.create_agent') as mock_create_agent:

        mock_chat_openai.return_value = mock_llm
        mock_get_prompt.return_value = mock_system_prompt

        # Import after patching
        from app.services.llm.agent import Agent

        # Create agent
        agent = Agent(vector_store=mock_vector_store)

        # Verify create_agent was called with response_format
        mock_create_agent.assert_called_once()
        call_kwargs = mock_create_agent.call_args.kwargs
        assert 'response_format' in call_kwargs
        assert call_kwargs['response_format'] == StructuredChatResponse
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_agent_structured_output.py -v`

Expected: Tests may pass or fail depending on current state - this establishes baseline. The key test is `test_agent_has_response_format_configured` which verifies the `response_format` parameter is passed.

**Step 3: Commit**

```bash
git add tests/test_agent_structured_output.py
git commit -m "test: add tests for structured output model and agent config"
```

---

## Task 4: Write Integration Test for Agent Structured Response

**Files:**
- Create: `tests/test_agent_integration_structured.py`

**Step 1: Write the integration test**

Create `tests/test_agent_integration_structured.py`:

```python
import pytest
from unittest.mock import MagicMock, patch
from app.services.llm.agent import Agent
from app.services.rag.vector_store import VectorStore


@pytest.mark.integration
def test_agent_invoke_returns_structured_response():
    """Test that agent invoke returns structured_response in result"""
    # This is an integration test - it will call the actual OpenAI API
    # Skip if OPENAI_API_KEY is not set
    import os
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    # Create real vector store (or use mock for faster tests)
    mock_vector_store = MagicMock(spec=VectorStore)

    # Mock the retriever tool
    mock_retriever_tool = MagicMock()
    mock_retriever_tool.name = "search_knowledge_base"

    with patch('app.services.llm.agent.get_system_prompt') as mock_get_prompt:
        mock_get_prompt.return_value = "You are a helpful assistant."

        agent = Agent(vector_store=mock_vector_store, retriever_tool=mock_retriever_tool)

        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Hello"}]},
            {}
        )

    # Assertions
    assert "structured_response" in result
    assert hasattr(result["structured_response"], "reply")
    assert hasattr(result["structured_response"], "confidence_score")
    assert hasattr(result["structured_response"], "escalate")
    assert isinstance(result["structured_response"].confidence_score, float)
    assert 0.0 <= result["structured_response"].confidence_score <= 1.0
    assert isinstance(result["structured_response"].escalate, bool)
```

**Step 2: Run test to see if it passes (may need OPENAI_API_KEY)**

Run: `poetry run pytest tests/test_agent_integration_structured.py -v`

Expected: May fail if OPENAI_API_KEY not set or if there are issues with the agent setup. This test validates end-to-end behavior.

**Step 3: Commit**

```bash
git add tests/test_agent_integration_structured.py
git commit -m "test: add integration test for agent structured response"
```

---

## Task 5: Update ChatService to Extract Structured Response

**Files:**
- Modify: `app/services/llm/chat.py`

**Step 1: Update the chat() method to extract from structured_response**

Replace the content of the `chat()` method (lines 23-56) with:

```python
def chat(self, message: str, conversation_history: list[dict], conversation_id: Optional[str] = None) -> ChatResponse:
    """Process a chat message using the RAG agent"""
    messages = conversation_history.copy()
    messages.append({"role": "user", "content": message})

    # Convert dict messages to LangChain Message objects
    lc_messages = []
    for msg in messages:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        else:
            lc_messages.append(AIMessage(content=msg["content"]))

    config = {
        "callbacks": [self.handler.handler],
        "metadata": {
            "langfuse_session_id": conversation_id or "default",
            "langfuse_tags": ["chat", "telco-agent"]
        }
    }

    result = self.agent.invoke({"messages": lc_messages}, config=config)

    # Extract structured response
    structured = result.get("structured_response")
    if structured is None:
        # Fallback handling if structured_response is missing
        return self._fallback_response(result)

    # Extract sources from tool calls
    sources = self._extract_sources(result)

    return ChatResponse(
        reply=structured.reply,
        escalate=structured.escalate,
        confidence_score=structured.confidence_score,
        sources=sources
    )
```

**Step 2: Commit**

```bash
git add app/services/llm/chat.py
git commit -m "feat: extract structured response from agent result"
```

---

## Task 6: Add _extract_sources Method to ChatService

**Files:**
- Modify: `app/services/llm/chat.py`

**Step 1: Add the _extract_sources method**

Add this method to the `ChatService` class (after `_should_escalate` method):

```python
def _extract_sources(self, result: dict) -> Optional[list[str]]:
    """Extract source document names from retriever tool calls"""
    sources = []
    messages = result.get("messages", [])

    for msg in messages:
        if hasattr(msg, "tool_calls"):
            for call in msg.tool_calls:
                if call.get("name") == "search_knowledge_base":
                    # Extract from tool args (query)
                    args = call.get("args", {})
                    # The source names are in the tool results, not args
                    # We need to look at ToolMessage responses
        elif hasattr(msg, "content") and isinstance(msg.content, list):
            # Tool results come back as content lists
            for content_item in msg.content:
                if isinstance(content_item, dict) and "text" in content_item:
                    # Parse source from formatted response: "[category - source]: content"
                    import re
                    text = content_item["text"]
                    # Match pattern like "[billing - billing_qna.json]: ..."
                    pattern = r'\[([^\]]+?-([^\]]+?))\]:'
                    matches = re.findall(pattern, text)
                    for match in matches:
                        if len(match) >= 2:
                            source = match[1]  # Second capture group is the source filename
                            if source not in sources:
                                sources.append(source)

    return sources if sources else None
```

**Step 2: Add re import at top of file**

Update imports to include `re`:

```python
import re
from pydantic import BaseModel
from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage
from .agent import Agent
from .callbacks import CallbackHandler
```

**Step 3: Commit**

```bash
git add app/services/llm/chat.py
git commit -m "feat: add _extract_sources method to parse sources from tool results"
```

---

## Task 7: Add _fallback_response Method to ChatService

**Files:**
- Modify: `app/services/llm/chat.py`

**Step 1: Add the _fallback_response method**

Add this method to the `ChatService` class:

```python
def _fallback_response(self, result: dict) -> ChatResponse:
    """Fallback response if structured_response is missing"""
    messages_list = result["messages"]
    last_message = messages_list[-1]
    reply = last_message.content if hasattr(last_message, 'content') else str(last_message)

    return ChatResponse(
        reply=reply,
        escalate=False,  # Default on fallback
        confidence_score=None,
        sources=self._extract_sources(result)
    )
```

**Step 2: Commit**

```bash
git add app/services/llm/chat.py
git commit -m "feat: add _fallback_response method for error handling"
```

---

## Task 8: Remove _should_escalate Method

**Files:**
- Modify: `app/services/llm/chat.py`

**Step 1: Remove the _should_escalate method**

Delete lines 58-65 (the `_should_escalate` method and its contents). This method is no longer needed since the LLM now determines the escalate flag directly.

**Step 2: Commit**

```bash
git add app/services/llm/chat.py
git commit -m "refactor: remove _should_escalate method (now handled by structured output)"
```

---

## Task 9: Write Tests for ChatService Changes

**Files:**
- Create: `tests/test_chat_service_structured.py`

**Step 1: Write the tests**

Create `tests/test_chat_service_structured.py`:

```python
import pytest
from unittest.mock import MagicMock, patch
from app.services.llm.chat import ChatService
from app.services.llm.agent import Agent
from app.services.llm.callbacks import CallbackHandler


def test_chat_service_extracts_structured_response():
    """Test that ChatService correctly extracts structured response"""
    # Mock agent
    mock_agent = MagicMock()
    mock_handler = MagicMock()

    # Create mock result with structured_response
    mock_result = {
        "messages": [],
        "structured_response": MagicMock(
            reply="Test reply",
            confidence_score=0.85,
            escalate=False
        )
    }
    mock_agent.invoke.return_value = mock_result

    # Create service
    service = ChatService(agent=mock_agent, handler=mock_handler)

    # Call chat
    response = service.chat(
        message="Test message",
        conversation_history=[],
        conversation_id="test-123"
    )

    # Assertions
    assert response.reply == "Test reply"
    assert response.confidence_score == 0.85
    assert response.escalate is False
    mock_agent.invoke.assert_called_once()


def test_chat_service_fallback_when_no_structured_response():
    """Test that ChatService uses fallback when structured_response is missing"""
    mock_agent = MagicMock()
    mock_handler = MagicMock()

    # Create mock result WITHOUT structured_response
    from langchain_core.messages import AIMessage
    mock_result = {
        "messages": [AIMessage(content="Fallback reply")]
    }
    mock_agent.invoke.return_value = mock_result

    service = ChatService(agent=mock_agent, handler=mock_handler)

    response = service.chat(
        message="Test message",
        conversation_history=[],
        conversation_id="test-123"
    )

    # Assertions - should use fallback
    assert response.reply == "Fallback reply"
    assert response.escalate is False  # Default in fallback
    assert response.confidence_score is None  # Not available in fallback


def test_chat_service_extracts_sources_from_tool_calls():
    """Test that _extract_sources correctly parses source names from tool results"""
    mock_agent = MagicMock()
    mock_handler = MagicMock()

    # Create mock result with tool results containing sources
    from langchain_core.messages import AIMessage, ToolMessage
    mock_result = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "search_knowledge_base",
                    "args": {"query": "test"}
                }]
            ),
            ToolMessage(
                content="[Billing - billing_qna.json]: Billing info here\n[Plans - plans_qna.json]: Plans info here",
                tool_call_id="test-id"
            )
        ],
        "structured_response": MagicMock(
            reply="Test reply",
            confidence_score=0.9,
            escalate=False
        )
    }
    mock_agent.invoke.return_value = mock_result

    service = ChatService(agent=mock_agent, handler=mock_handler)

    response = service.chat(
        message="Test message",
        conversation_history=[],
        conversation_id="test-123"
    )

    # Should extract sources from tool results
    assert response.sources is not None
    assert "billing_qna.json" in response.sources
    assert "plans_qna.json" in response.sources
```

**Step 2: Run tests to verify they pass**

Run: `poetry run pytest tests/test_chat_service_structured.py -v`

Expected: All tests should pass

**Step 3: Commit**

```bash
git add tests/test_chat_service_structured.py
git commit -m "test: add tests for ChatService structured response handling"
```

---

## Task 10: Write API Integration Test

**Files:**
- Create: `tests/test_api_structured_output.py`

**Step 1: Write the API integration test**

Create `tests/test_api_structured_output.py`:

```python
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.mark.integration
def test_chat_endpoint_returns_structured_fields():
    """Test that /chat endpoint returns reply, escalate, and confidence_score"""
    import os
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "message": "What is my current bill?",
            "conversation_id": "test-123"
        }
    )

    assert response.status_code == 200

    data = response.json()
    assert "reply" in data
    assert "escalate" in data
    assert "confidence_score" in data

    # Verify types
    assert isinstance(data["reply"], str)
    assert isinstance(data["escalate"], bool)
    assert isinstance(data["confidence_score"], float) or data["confidence_score"] is None

    # Verify confidence_score range if present
    if data["confidence_score"] is not None:
        assert 0.0 <= data["confidence_score"] <= 1.0


@pytest.mark.integration
def test_chat_endpoint_with_triggers_escalation():
    """Test that LLM can set escalate=True when appropriate"""
    import os
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    client = TestClient(app)

    # This question should trigger escalation
    response = client.post(
        "/chat",
        json={
            "message": "I need to speak to a human agent immediately about a legal matter",
            "conversation_id": "test-456"
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert "escalate" in data
    # The LLM should recognize this needs escalation
    # Note: This depends on the LLM's judgment, so we just check the field exists
```

**Step 2: Run the API integration test**

Run: `poetry run pytest tests/test_api_structured_output.py -v`

Expected: Tests should pass if OPENAI_API_KEY is set

**Step 3: Commit**

```bash
git add tests/test_api_structured_output.py
git commit -m "test: add API integration tests for structured output"
```

---

## Task 11: Update System Prompt in Langfuse

**Files:**
- External: Langfuse Prompt Management (https://langfuse.com)

**Step 1: Access the system prompt**

1. Go to https://cloud.langfuse.com
2. Navigate to Prompts
3. Find the telco-agent system prompt

**Step 2: Update the prompt with structured output instructions**

Add these instructions to the system prompt:

```
When responding to users, you must provide a structured response with the following fields:

1. **reply**: A helpful, natural language response to the user's question based on the retrieved knowledge base information.

2. **confidence_score**: A float between 0.0 and 1.0 indicating your confidence in the answer:
   - 0.9-1.0: High confidence - answer comes directly from retrieved knowledge base with clear, relevant information
   - 0.7-0.9: Good confidence - answer is based on retrieved information but may require some inference
   - 0.5-0.7: Moderate confidence - answer is partially supported by retrieved information but has gaps
   - 0.3-0.5: Low confidence - limited relevant information found, answer may be incomplete
   - 0.0-0.3: Very low confidence - little to no relevant information found

3. **escalate**: A boolean flag (true/false) indicating whether to escalate to a human agent:
   - Set to true if:
     * The user's question cannot be answered with available information
     * The request requires capabilities outside your scope (e.g., account changes, refunds, cancellations)
     * The user explicitly asks for a human agent
     * The situation involves sensitive issues (legal, fraud, billing disputes)
   - Set to false if:
     * You can provide a helpful answer from the knowledge base
     * The request is for general information or guidance

IMPORTANT: Always search the knowledge base first using the search_knowledge_base tool before determining your confidence score or escalation need.
```

**Step 3: Commit a note about the prompt update**

Create a file to document the prompt update:

```bash
echo "# System Prompt Update for Structured Output

Date: $(date +%Y-%m-%d)

The system prompt in Langfuse has been updated to include instructions for:

1. Confidence scoring guidelines (0.0-1.0 range with descriptions)
2. Escalation criteria (when to set escalate=true)

This enables the LLM to properly populate the structured response fields.
" > docs/system-prompt-structured-output.md
```

```bash
git add docs/system-prompt-structured-output.md
git commit -m "docs: document system prompt update for structured output"
```

---

## Task 12: Run All Tests and Verify

**Step 1: Run all tests**

```bash
poetry run pytest -v
```

Expected: All tests should pass

**Step 2: Run specific structured output tests**

```bash
poetry run pytest tests/test_agent_structured_output.py tests/test_chat_service_structured.py tests/test_api_structured_output.py -v
```

Expected: All structured output tests should pass

**Step 3: Run with coverage**

```bash
poetry run pytest --cov=app --cov-report=html
```

Expected: Coverage report shows new code is tested

**Step 4: Manual testing with the API**

Start the server:

```bash
poetry run python run.py
```

Test with curl:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is my current bill?", "conversation_id": "test-123"}'
```

Expected response should include `reply`, `confidence_score`, `escalate`, and optionally `sources`.

**Step 5: Commit any fixes**

If tests fail and fixes are needed:

```bash
git add ...
git commit -m "fix: ..."
```

---

## Task 13: Update Documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md` (if exists)

**Step 1: Update CLAUDE.md with structured output information**

Add to the "Architecture Overview" section in `CLAUDE.md`:

```markdown
### Structured Output

The agent uses LangChain's `response_format` parameter to return structured responses:

**StructuredChatResponse Model:**
- `reply`: Natural language response
- `confidence_score`: Float (0.0-1.0) indicating answer confidence
- `escalate`: Boolean flag for human escalation

The LLM determines these values directly based on the system prompt instructions.
```

**Step 2: Update the service architecture section**

Update the "Service Architecture" section to reflect the changes:

```markdown
**LLM Services (`app/services/llm/`)**
- `agent.py` - `Agent` class with `StructuredChatResponse` model for structured output
- `callbacks.py` - `CallbackHandler` class wrapping Langfuse tracing
- `chat.py` - `ChatService` class for orchestration, extracts structured response from agent
```

**Step 3: Update the "Escalation Logic" section**

Replace the existing "Escalation Logic" section with:

```markdown
### Escalation Logic

The LLM now directly determines the `escalate` flag based on the system prompt instructions. The escalation criteria are:

- User question cannot be answered with available information
- Request requires capabilities outside agent scope (account changes, refunds)
- User explicitly asks for a human agent
- Sensitive issues (legal, fraud, billing disputes)

The system prompt in Langfuse contains detailed guidelines for when to set `escalate=true`.
```

**Step 4: Commit documentation updates**

```bash
git add CLAUDE.md README.md
git commit -m "docs: update documentation for structured output feature"
```

---

## Summary

After completing all tasks, the system will:

1. Use GPT-4o's native structured output via `response_format`
2. Return validated responses with `reply`, `confidence_score`, and `escalate`
3. Eliminate keyword-based escalation detection
4. Extract sources from tool call results
5. Provide fallback behavior if structured output fails

### Files Modified

- `app/services/llm/agent.py` - Added `StructuredChatResponse` model and `response_format` parameter
- `app/services/llm/chat.py` - Updated to extract structured response, added `_extract_sources` and `_fallback_response`, removed `_should_escalate`

### Files Created

- `tests/test_agent_structured_output.py` - Unit tests for model and agent config
- `tests/test_agent_integration_structured.py` - Integration test for agent
- `tests/test_chat_service_structured.py` - Tests for ChatService changes
- `tests/test_api_structured_output.py` - API integration tests
- `docs/system-prompt-structured-output.md` - Documentation of prompt update

### External Changes

- Langfuse system prompt updated with structured output instructions
