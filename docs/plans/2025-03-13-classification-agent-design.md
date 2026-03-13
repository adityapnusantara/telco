# Classification Agent Design

## Overview

Replace the current `ChatOpenAI` + manual JSON parsing approach for reply classification with a `create_agent` based solution using `response_format` for automatic structured output.

## Motivation

The current implementation in `ChatService._classify_reply()`:
- Uses `ChatOpenAI` directly
- Manually constructs prompt asking for JSON
- Manually parses JSON with `json.loads()`
- Has try-catch for JSON parsing errors
- Falls back to heuristics on error

**Goal:** Cleaner code without manual JSON parsing and error handling.

## Architecture

```
ChatService.__init__() - Initialize classification agent
    ↓
ChatService._classify_reply(reply, has_sources)
    ↓
Compile prompt with variables (reply, context)
    ↓
ClassificationAgent.invoke(prompt)
    ↓
OpenAI LLM (gpt-4o) dengan structured output
    ↓
ReplyClassification(confidence_score=0.85, escalate=False)
```

## Components

### Langfuse Prompt Management

**Prompt name:** `telco-customer-service-classification`

**Template:**
```
Analyze this customer service reply:

Reply: {{reply}}

Context: {{context}}

Classify and return confidence_score (0.0-1.0) and escalate (true/false).
```

**Config:**
```json
{
    "model": "gpt-4o",
    "temperature": 0
}
```

### File Changes

**`app/prompts/langfuse.py`** - Add new functions:
```python
def get_classification_prompt_obj():
    """Get classification prompt object from Langfuse for .compile()"""
    return langfuse_client.get_prompt("telco-customer-service-classification")

def get_classification_config() -> dict:
    """Get classification model config from Langfuse"""
    config = langfuse_client.get_prompt("telco-customer-service-classification")
    return config.config
```

**`app/services/llm/chat.py`** - Update `ChatService`:

1. **Update `__init__()`**:
   - Initialize `self._classification_prompt_obj`
   - Create `self._classification_agent` with `create_agent()`

2. **Update `_classify_reply()`**:
   - Use `self._classification_prompt_obj.compile()`
   - Remove manual JSON parsing
   - Return `result["structured_response"]` directly

## Implementation Details

### ChatService.__init__()

```python
def __init__(self, agent: Agent, handler: CallbackHandler):
    self.agent = agent
    self.handler = handler

    # Get prompt & config from Langfuse (initialize once)
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

### ChatService._classify_reply()

```python
async def _classify_reply(self, reply: str, has_sources: bool) -> ReplyClassification:
    """Classify reply using agent with structured output"""
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

## Data Flow

1. **Initialization** (`__init__`):
   - Fetch prompt object from Langfuse: `get_classification_prompt_obj()`
   - Fetch model config: `get_classification_config()`
   - Create classification agent with `create_agent()`

2. **Classification** (`_classify_reply`):
   - Compile prompt with `reply` and `context` variables
   - Invoke classification agent with Langfuse callback
   - Extract `result["structured_response"]`
   - Return `ReplyClassification` object

## Benefits

- ✅ Cleaner code - no manual JSON parsing
- ✅ Type-safe - Pydantic validation automatic
- ✅ Consistent with Agent class pattern
- ✅ Langfuse tracing for all classification calls
- ✅ Managed prompt in Langfuse (easy to update)
- ✅ No error handling needed for JSON parsing
- ✅ Prompt object initialized once in `__init__` (more efficient)

## Testing

Update existing tests in `tests/test_chat_service.py`:
- Mock `get_classification_prompt_obj()` instead of `_classification_llm.ainvoke`
- Verify `result["structured_response"]` is returned correctly
