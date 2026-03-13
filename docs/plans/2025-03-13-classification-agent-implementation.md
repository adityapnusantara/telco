# Classification Agent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace ChatOpenAI + manual JSON parsing with create_agent + response_format for cleaner classification code.

**Architecture:** Use LangChain's `create_agent` with empty tools list and `response_format=ReplyClassification` to get automatic structured output without JSON parsing. Prompt and config managed in Langfuse.

**Tech Stack:** LangChain v1.2.11, OpenAI gpt-4o, Langfuse prompt management

---

## Task 1: Add Langfuse Functions for Classification Prompt

**Files:**
- Modify: `app/prompts/langfuse.py`

**Step 1: Add get_classification_prompt_obj() function**

```python
def get_classification_prompt_obj():
    """Get classification prompt object from Langfuse for .compile()

    Returns prompt object with template containing {{reply}} and {{context}} variables.
    """
    return langfuse_client.get_prompt("telco-customer-service-classification")
```

**Step 2: Add get_classification_config() function**

```python
def get_classification_config() -> dict:
    """Get classification model config from Langfuse.

    Returns: {"model": "gpt-4o", "temperature": 0}
    """
    config = langfuse_client.get_prompt("telco-customer-service-classification")
    return config.config
```

**Step 3: Write tests for new functions**

Create test file: `tests/test_langfuse_classification.py`

```python
import pytest
from unittest.mock import patch, MagicMock
from app.prompts.langfuse import get_classification_prompt_obj, get_classification_config


def test_get_classification_prompt_obj():
    """Test getting classification prompt object from Langfuse"""
    mock_prompt_obj = MagicMock()
    mock_prompt_obj.compile.return_value = "Compiled prompt"

    with patch('app.prompts.langfuse.langfuse_client.get_prompt') as mock_get_prompt:
        mock_get_prompt.return_value = mock_prompt_obj

        result = get_classification_prompt_obj()

        mock_get_prompt.assert_called_once_with("telco-customer-service-classification")
        assert result == mock_prompt_obj


def test_get_classification_config():
    """Test getting classification config from Langfuse"""
    mock_prompt_obj = MagicMock()
    mock_prompt_obj.config = {"model": "gpt-4o", "temperature": 0}

    with patch('app.prompts.langfuse.langfuse_client.get_prompt') as mock_get_prompt:
        mock_get_prompt.return_value = mock_prompt_obj

        result = get_classification_config()

        mock_get_prompt.assert_called_once_with("telco-customer-service-classification")
        assert result == {"model": "gpt-4o", "temperature": 0}
```

**Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_langfuse_classification.py -v
```

Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add app/prompts/langfuse.py tests/test_langfuse_classification.py
git commit -m "feat: add Langfuse functions for classification prompt"
```

---

## Task 2: Update ChatService.__init__() to Use Classification Agent

**Files:**
- Modify: `app/services/llm/chat.py:34-42`
- Test: `tests/test_chat_service.py`

**Step 1: Update imports**

```python
from app.prompts.langfuse import get_system_prompt, get_model_config, get_classification_prompt_obj, get_classification_config
```

**Step 2: Replace _classification_llm with _classification_agent in __init__()**

Find this code (lines 38-42):
```python
# Classification LLM (gpt-4o-mini for cost/speed)
self._classification_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)
```

Replace with:
```python
# Classification Agent - create_agent with empty tools list
self._classification_prompt_obj = get_classification_prompt_obj()
model_config = get_classification_config()

classification_llm = ChatOpenAI(
    model=model_config["model"],
    temperature=model_config["temperature"]
)

self._classification_agent = create_agent(
    model=classification_llm,
    tools=[],  # Empty - no tools needed for classification
    system_prompt=self._classification_prompt_obj,
    response_format=ReplyClassification
)
```

**Step 3: Add create_agent import**

Add to imports at top of file:
```python
from langchain.agents import create_agent
```

**Step 4: Update test for ChatService.__init__()**

In `tests/test_chat_service.py`, update `test_chat_service_init`:

```python
def test_chat_service_init():
    """Test ChatService class initialization"""
    mock_agent = Mock(spec=Agent)
    mock_handler = Mock(spec=CallbackHandler)

    with patch('app.services.llm.chat.get_classification_prompt_obj') as mock_get_prompt, \
         patch('app.services.llm.chat.get_classification_config') as mock_get_config, \
         patch('app.services.llm.chat.ChatOpenAI') as mock_chat_openai, \
         patch('app.services.llm.chat.create_agent') as mock_create_agent:

        mock_prompt_obj = MagicMock()
        mock_get_prompt.return_value = mock_prompt_obj
        mock_get_config.return_value = {"model": "gpt-4o", "temperature": 0}
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        mock_agent_instance = MagicMock()
        mock_create_agent.return_value = mock_agent_instance

        service = ChatService(agent=mock_agent, handler=mock_handler)

        assert service.agent == mock_agent
        assert service.handler == mock_handler
        assert service._classification_agent == mock_agent_instance
        assert service._classification_prompt_obj == mock_prompt_obj
```

**Step 5: Run tests to verify they pass**

```bash
poetry run pytest tests/test_chat_service.py::test_chat_service_init -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add app/services/llm/chat.py tests/test_chat_service.py
git commit -m "refactor: replace ChatOpenAI with create_agent for classification"
```

---

## Task 3: Update _classify_reply() Method

**Files:**
- Modify: `app/services/llm/chat.py:44-77`
- Test: `tests/test_chat_service.py`

**Step 1: Write the updated test first**

Add to `tests/test_chat_service.py`:

```python
@pytest.mark.asyncio
async def test_classify_reply_uses_agent_with_structured_output():
    """Test _classify_reply uses classification agent with structured output"""
    mock_agent = Mock()
    mock_handler = Mock()

    service = ChatService(agent=mock_agent, handler=mock_handler)

    # Mock the classification agent to return structured_response
    mock_result = {
        "structured_response": ReplyClassification(
            confidence_score=0.85,
            escalate=False
        )
    }

    with patch.object(service._classification_prompt_obj, 'compile') as mock_compile, \
         patch.object(service._classification_agent, 'invoke', return_value=mock_result) as mock_invoke:

        mock_compile.return_value = "Compiled prompt"

        classification = await service._classify_reply(
            "You can check your balance in the app.",
            has_sources=True
        )

        # Verify prompt was compiled with correct variables
        mock_compile.assert_called_once_with(
            reply="You can check your balance in the app.",
            context="Sources found from knowledge base"
        )

        # Verify agent was invoked with correct config
        mock_invoke.assert_called_once()
        call_args = mock_invoke.call_args
        assert call_args[1]["callbacks"] == [service.handler.handler]
        assert "classification" in call_args[1]["metadata"]["langfuse_tags"]

        # Verify structured response was returned
        assert classification.confidence_score == 0.85
        assert classification.escalate is False
```

**Step 2: Run test to verify it fails**

```bash
poetry run pytest tests/test_chat_service.py::test_classify_reply_uses_agent_with_structured_output -v
```

Expected: FAIL (method still uses old ChatOpenAI approach)

**Step 3: Implement _classify_reply() with agent**

Replace entire `_classify_reply` method (lines 44-77) with:

```python
async def _classify_reply(self, reply: str, has_sources: bool) -> ReplyClassification:
    """Classify reply to extract confidence_score and escalate using LLM.

    Args:
        reply: The agent's natural language response
        has_sources: Whether sources were found in the knowledge base

    Returns:
        ReplyClassification with confidence_score and escalate
    """
    # Compile prompt with variables
    context = 'Sources found from knowledge base' if has_sources else 'No sources available from knowledge base'
    compiled_prompt = self._classification_prompt_obj.compile(
        reply=reply,
        context=context
    )

    result = self._classification_agent.invoke(
        {"messages": [{"role": "user", "content": compiled_prompt}]},
        config={
            "callbacks": [self.handler.handler],
            "metadata": {
                "langfuse_tags": ["classification", "metadata-extraction"],
                "classification_task": "reply_metadata"
            }
        }
    )

    return result["structured_response"]
```

**Step 4: Run test to verify it passes**

```bash
poetry run pytest tests/test_chat_service.py::test_classify_reply_uses_agent_with_structured_output -v
```

Expected: PASS

**Step 5: Remove obsolete test**

Delete the old test `test_classify_reply_fallback_on_error` from `tests/test_chat_service.py` (if it exists) as fallback is no longer needed with structured output.

**Step 6: Run all chat_service tests to verify nothing broke**

```bash
poetry run pytest tests/test_chat_service.py -v
```

Expected: All PASS

**Step 7: Commit**

```bash
git add app/services/llm/chat.py tests/test_chat_service.py
git commit -m "refactor: _classify_reply uses agent with structured output"
```

---

## Task 4: Remove Unused Imports

**Files:**
- Modify: `app/services/llm/chat.py:1-12`

**Step 1: Check for unused imports**

Review imports at top of `chat.py`. The `json` import may no longer be needed since we removed manual JSON parsing.

**Step 2: Remove json import if no longer used**

If `json` is only used in the old `_classify_reply`, remove it from imports:
```python
import asyncio
# import json  # Remove this line
import logging
import re
```

**Step 3: Run tests to verify nothing broke**

```bash
poetry run pytest tests/test_chat_service.py -v
```

Expected: All PASS

**Step 4: Commit**

```bash
git add app/services/llm/chat.py
git commit -m "refactor: remove unused json import"
```

---

## Task 5: Integration Testing

**Files:**
- Test: Manual verification

**Step 1: Start the server**

```bash
poetry run python run.py
```

**Step 2: Test non-streaming endpoint**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How do I check my balance?",
    "conversation_history": []
  }'
```

Expected: Response with `confidence_score` and `escalate` fields populated

**Step 3: Test streaming endpoint**

```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello",
    "conversation_history": []
  }'
```

Expected: Token events followed by end event with classified metadata

**Step 4: Check Langfuse**

Go to Langfuse dashboard and verify:
- Classification calls are traced with "classification" tag
- Prompt "telco-customer-service-classification" is logged
- Variables `reply` and `context` are visible

**Step 5: Run full test suite**

```bash
poetry run pytest tests/ -v
```

Expected: All tests PASS (44+ tests)

---

## Task 6: Documentation Update

**Files:**
- Update: `CLAUDE.md` (if needed)

**Step 1: Review CLAUDE.md**

Check if the classification approach is documented. Update if reference to old ChatOpenAI approach exists.

**Step 2: Add classification agent to architecture docs (if applicable)**

Add note about classification agent using `create_agent` with empty tools list.

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update classification agent documentation"
```

---

## Verification Checklist

- [ ] All new tests pass
- [ ] All existing tests still pass
- [ ] Manual testing of `/chat` endpoint works
- [ ] Manual testing of `/chat/stream` endpoint works
- [ ] Langfuse traces show classification calls with proper tags
- [ ] No unused imports remain
- [ ] Code follows existing patterns (DRY, YAGNI)
