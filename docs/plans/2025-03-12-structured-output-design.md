# Design: Structured Output with create_agent

**Date:** 2025-03-12
**Status:** Approved
**Author:** Claude + User

## Overview

Add `response_format` to LangChain's `create_agent` to return structured responses with `reply`, `confidence_score`, and `escalate` fields directly from the LLM, eliminating post-processing and keyword-based escalation detection.

## Requirements

- Use GPT-4o's native structured output capability
- LLM determines `confidence_score` (0-1) based on its certainty
- LLM determines `escalate` flag directly
- `sources` field continues to be populated by ChatService from retriever tool calls

## Architecture

```
User Request → FastAPI /chat → ChatService
    ↓
Agent.invoke() with response_format
    ↓
LangChain Agent → GPT-4o (native structured output)
    ↓
Result with structured_response key
    ↓
ChatService extracts structured fields + sources from tool calls
    ↓
ChatResponse(reply, escalate, confidence_score, sources)
```

## Data Model

### New Model: StructuredChatResponse

```python
from pydantic import BaseModel, Field

class StructuredChatResponse(BaseModel):
    """Structured response schema for agent output"""
    reply: str = Field(
        description="The natural language response to the user's question"
    )
    confidence_score: float = Field(
        description="Confidence score from 0.0 to 1.0 indicating how certain the agent is about its answer",
        ge=0.0,
        le=1.0
    )
    escalate: bool = Field(
        description="True if the user should be escalated to a human agent"
    )
```

### Existing Models (No Changes)

- `ChatRequest` - unchanged
- `ChatResponse` - already has required fields

## Implementation Details

### 1. Agent Class Changes (`app/services/llm/agent.py`)

Add `StructuredChatResponse` model and pass to `create_agent`:

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from app.prompts.langfuse import get_system_prompt
from app.services.rag.retriever import RetrieverTool
from app.services.rag.vector_store import VectorStore


class StructuredChatResponse(BaseModel):
    """Structured response schema for agent output"""
    reply: str = Field(description="The natural language response to the user's question")
    confidence_score: float = Field(
        description="Confidence score from 0.0 to 1.0 indicating how certain the agent is about its answer",
        ge=0.0,
        le=1.0
    )
    escalate: bool = Field(description="True if the user should be escalated to a human agent")


class Agent:
    """LangChain agent wrapper with explicit initialization and DI"""

    def __init__(self, vector_store: VectorStore, retriever_tool=None):
        self._vector_store = vector_store
        self._llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self._system_prompt = get_system_prompt()

        if retriever_tool:
            self._retriever_tool = retriever_tool.tool
        else:
            self._retriever_tool = RetrieverTool(vector_store).tool

        self._agent = create_agent(
            model=self._llm,
            tools=[self._retriever_tool],
            system_prompt=self._system_prompt,
            response_format=StructuredChatResponse  # NEW
        )

    def invoke(self, messages, config):
        """Invoke the agent with messages and config"""
        return self._agent.invoke(messages, config)
```

### 2. ChatService Changes (`app/services/llm/chat.py`)

Update `chat()` method and add `_extract_sources()`. Remove `_should_escalate()`.

```python
from pydantic import BaseModel
from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage
from .agent import Agent
from .callbacks import CallbackHandler


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

    def _extract_sources(self, result: dict) -> Optional[list[str]]:
        """Extract source document names from retriever tool calls"""
        sources = []
        messages = result.get("messages", [])

        for msg in messages:
            if hasattr(msg, "tool_calls"):
                for call in msg.tool_calls:
                    if call.get("name") == "search_knowledge_base":
                        # Extract source metadata from tool results
                        # Implementation depends on tool result format
                        pass

        return sources if sources else None

    def _fallback_response(self, result: dict) -> ChatResponse:
        """Fallback response if structured_response is missing"""
        messages_list = result["messages"]
        last_message = messages_list[-1]
        reply = last_message.content

        return ChatResponse(
            reply=reply,
            escalate=False,  # Default on fallback
            confidence_score=None,
            sources=self._extract_sources(result)
        )
```

## Error Handling

| Scenario | Handling |
|----------|----------|
| `structured_response` missing | Fallback to parsing last message content |
| Pydantic validation error | Log error, return fallback response |
| No retriever tool calls | Set `sources = None` |
| Tool returned no results | Set `sources = []` |

## Testing

### Unit Tests (`tests/test_agent_structured_output.py`)

- Test agent returns `structured_response` in result
- Test all required fields present (reply, confidence_score, escalate)
- Test confidence_score is within valid range (0-1)

### Integration Tests (`tests/test_chat_structured_output.py`)

- Test `/chat` endpoint returns all fields
- Test escalate flag works correctly
- Test confidence_score is a valid float

## Migration Steps

1. Add `StructuredChatResponse` model to `agent.py`
2. Update `Agent.__init__` to pass `response_format`
3. Update `ChatService.chat()` to extract from `structured_response`
4. Add `_extract_sources()` method to `ChatService`
5. Add `_fallback_response()` method for error handling
6. Remove `_should_escalate()` method
7. Write unit and integration tests
8. Update system prompt in Langfuse to instruct on confidence scoring and escalation
9. Run tests and verify

## System Prompt Updates

The system prompt in Langfuse should be updated to instruct the LLM to:

1. **Confidence scoring**: Set higher scores (0.8-1.0) when answers come directly from retrieved knowledge base with clear context. Set lower scores (0.3-0.6) when information is incomplete or requires inference.

2. **Escalation**: Set `escalate=True` only when:
   - The user's question cannot be answered with available information
   - The request requires capabilities outside the agent's scope (e.g., account changes, refunds)
   - The user explicitly asks for a human agent

## Open Questions

- [ ] Exact format of tool call results for sources extraction (needs inspection of actual response format)
- [ ] Whether to log when fallback response is used
