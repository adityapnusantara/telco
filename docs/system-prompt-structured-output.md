# System Prompt Update for Structured Output

**Date:** 2026-03-12

## Overview

The system prompt in Langfuse Prompt Management needs to be manually updated to include instructions for structured output fields.

## Manual Update Required

**Location:** https://cloud.langfuse.com
**Prompt Name:** `telco-customer-service-agent`

## Instructions to Add

Add the following instructions to the system prompt:

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

## Structured Output Schema

The response model expects the following structure:

```python
class ChatResponse(BaseModel):
    reply: str                    # Agent's response
    escalate: bool                # Whether to escalate to human
    sources: Optional[List[str]]  # Retrieved document names
    confidence_score: Optional[float]  # Confidence score (0-1)
```

## Why This Is Needed

The system prompt instructions enable the LLM to properly populate the structured response fields that were introduced in the structured output feature branch. Without these instructions, the LLM may not understand how to properly set confidence scores or escalation flags.

## Steps to Update

1. Go to https://cloud.langfuse.com
2. Navigate to **Prompts** section
3. Find the `telco-customer-service-agent` prompt
4. Edit the prompt and append the structured output instructions above
5. Save and publish the updated prompt

## Related Code

- `app/api/models.py` - ChatResponse model with structured fields
- `app/services/llm/chat.py` - ChatService that extracts structured response
- `app/services/llm/agent.py` - Agent with structured output support
