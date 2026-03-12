# Telco Customer Service AI Agent - Design Document

**Date**: 2025-03-11
**Author**: AI Engineer Assignment
**Status**: Design Approved

## Overview

This document outlines the design for a Customer Service AI Agent for a telecommunications company. The agent handles inbound customer questions via chat, answering questions about service plans, billing, and troubleshooting - with escalation to human agents when needed.

**Assignment Scope**: Question 1 - Build working AI agent with RAG pipeline
**Future Extension**: Question 2 - Production system design and evaluation

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           FastAPI /chat                             │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   LangChain create_agent                            │
│  - System prompt from Langfuse Prompt Management                    │
│  - Tool-calling agent with RAG retriever                            │
│  - CallbackHandler for tracing                                      │
└─────────────────────────────────────────────────────────────────────┘
                    │           │           │
                    ▼           ▼           ▼
           ┌──────────┐  ┌─────────┐  ┌────────────┐
           │Retriever │  │ChatOpenAI│  │Langfuse    │
           │Tool      │  │(GPT-4o)  │  │Callback    │
           └──────────┘  └─────────┘  └────────────┘
                │                            ▲
                ▼                            │
    ┌─────────────────────┐                  │
    │   QdrantVectorStore │                  │
    │   (Cloud)           │                  │
    └─────────────────────┘                  │
                                                │
        ┌───────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Langfuse                                     │
│  ┌─────────────────┐     ┌─────────────────┐                       │
│  │Prompt Management│     │   Tracing &     │                       │
│  │                 │     │   Observability │                       │
│  │- System prompts │     │   - Traces      │                       │
│  │- Version control│     │   - Spans       │                       │
│  │- Labels (prod)  │     │   - Scores      │                       │
│  └─────────────────┘     └─────────────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## API Design

### Endpoint

**`POST /chat`**

### Request Model

```python
class ChatRequest(BaseModel):
    message: str                           # Current user message
    session_id: str | None = None     # For session tracking (Q2)
    conversation_history: list[dict] | None = None  # Previous messages
```

### Response Model

```python
class ChatResponse(BaseModel):
    reply: str                             # Agent's response
    escalate: bool                         # Whether to escalate to human
    sources: list[str] | None = None       # Retrieved doc names
    confidence_score: float | None = None  # Optional confidence metric
```

### Response Codes

- `200 OK`: Successful response
- `400 Bad Request`: Invalid input
- `500 Internal Server Error`: LLM/vector store failures

---

## Tech Stack

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | ^0.135.1 | Web framework |
| uvicorn | ^0.41.0 | ASGI server |
| pydantic | ^2.12.5 | Data validation |
| langchain | ^1.2.11 | LLM framework |
| langchain-openai | ^1.1.11 | OpenAI integration |
| langchain-qdrant | ^1.1.0 | Vector store |
| langchain-text-splitters | ^1.1.1 | Text processing |
| langfuse | ^4.0.0 | Prompt management & tracing |
| qdrant-client | ^1.17.0 | Qdrant client |

### Environment Variables

```bash
# OpenAI
OPENAI_API_KEY=sk-proj-...

# Langfuse
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com

# Qdrant Cloud
QDRANT_URL=https://cfcd371f-c4ef-4a64-99c3-243b273a5f07.europe-west3-0.gcp.cloud.qdrant.io
QDRANT_API_KEY=...

# Application
APP_ENV=development
LOG_LEVEL=info
```

---

## RAG Pipeline Design

### Chunking Strategy: Q&A Pairs

**Approach**: Pre-process documents into question-answer pairs

**Why Q&A Pairs**:
- Customer queries are naturally question-based
- Better semantic matching for retrieval
- Each chunk is self-contained
- No broken context at boundaries

**Document Structure**:
```python
class QNADocument(BaseModel):
    question: str
    answer: str
    source: str
    category: str  # "billing", "plans", "troubleshooting"
```

### Embedding Model

| Choice | Model | Reasoning |
|--------|-------|-----------|
| Selected | `text-embedding-3-small` | Cost-effective, fast, good quality |
| Dimensions | 512 | Smaller vectors = faster search |

### Vector Store

- **Provider**: Qdrant Cloud
- **Collection**: `telco_knowledge_base`
- **Distance**: Cosine

---

## System Prompt Design

**Prompt Name**: `telco-customer-service-agent`
**Type**: `chat`
**Managed in**: Langfuse Prompt Management

### Prompt Structure

```yaml
role: system
content: |
  You are a customer service agent for {{company_name}}, a telecommunications provider.

  Your capabilities:
  - Answer questions about billing, service plans, and troubleshooting
  - Use ONLY information from the retrieved knowledge base
  - If the information is not in the knowledge base, acknowledge it honestly

  Escalation rules:
  - Escalate to human if you cannot confidently answer from the knowledge base
  - DO NOT make up or hallucinate information
  - Clearly state when you don't know something

  Tone: Professional, helpful, concise

  When to escalate:
  - No relevant information found in knowledge base
  - Customer requests to speak to human
  - Complex billing disputes requiring manual review

  Escalation contact: {{escalation_contact}}
```

**Variables**:
- `{{company_name}}` - "MyTelco" (configurable)
- `{{escalation_contact}}` - Human agent contact info

---

## Data Flow

```
User Request
      │
      ▼
┌─────────────────────────────────────────────┐
│  1. Initialize Langfuse CallbackHandler      │
│     - Set session_id, user_id               │
│     - Set tags for request type              │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│  2. Fetch System Prompt from Langfuse       │
│     - Get "telco-customer-service-agent"    │
│     - Label: "production"                    │
│     - Type: "chat"                           │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│  3. Create Agent with Tools                  │
│     - create_agent(model, tools, prompt)     │
│     - Tools: [retriever_tool]                │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│  4. Invoke Agent                            │
│     - Pass user message + history            │
│     - Include CallbackHandler                │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│  5. Agent Decision Logic                    │
│  - Retrieval Success? → Use/escalate         │
│  - Confidence Check → Low = escalate        │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│  6. Extract Response                        │
│     - reply, escalate, sources, confidence   │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│  7. Log to Langfuse (auto-flush)            │
└─────────────────────────────────────────────┘
      │
      ▼
   Return ChatResponse
```

---

## Escalation Logic

### Triggers

| Trigger | Condition | Action |
|---------|-----------|--------|
| No retrieval | `retrieved_docs` is empty | `escalate=True` |
| Low confidence | `score < 0.5` threshold | `escalate=True` |
| Agent uncertainty | Agent asks for human help | `escalate=True` |

### Error Handling

| Error Type | Handling |
|------------|----------|
| Langfuse API timeout | Use fallback system prompt, log warning |
| Qdrant connection fail | Return error, `escalate=True` |
| OpenAI rate limit | Retry with exponential backoff |
| Empty retrieval | Acknowledge inability, `escalate=True` |

---

## Project Structure

```
telco/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI entry point
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   └── chat.py         # /chat endpoint
│   │   └── models/
│   │       ├── __init__.py
│   │       └── requests.py     # Pydantic models
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py           # Configuration management
│   │   └── logging.py          # Logging setup
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm/
│   │   │   ├── __init__.py
│   │   │   └── agent.py        # LangChain agent orchestration
│   │   └── rag/
│   │       ├── __init__.py
│   │       ├── retriever.py    # RAG retrieval logic
│   │       └── knowledge_base.py # Q&A document management
│   └── prompts/
│       ├── __init__.py
│       └── langfuse.py         # Langfuse prompt fetcher
├── data/
│   └── kb/                     # Knowledge base Q&A pairs
│       ├── billing_qna.json
│       ├── plans_qna.json
│       └── troubleshooting_qna.json
├── docs/
│   ├── design/                 # Design documents
│   └── diagrams/               # Architecture diagrams
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_rag.py
│   └── test_llm.py
├── .env.example
├── pyproject.toml
├── Dockerfile
└── README.md
```

---

## Testing Strategy

### Test Categories

| Test Type | What It Tests | Tools |
|-----------|---------------|-------|
| Unit | Individual services | pytest, unittest.mock |
| Integration | API endpoint + services | pytest, httpx |
| E2E | Full request → response | pytest, test client |

### Key Test Scenarios

- ✅ Successful chat response with retrieval
- ✅ Escalation when no relevant docs found
- ✅ Conversation history handling
- ✅ Langfuse callback integration
- ✅ Error handling for external service failures

---

## Limitations & Production Improvements

### Current Limitations

1. **Manual Q&A creation**: Documents are manually converted to Q&A pairs
2. **Single collection**: All documents in one Qdrant collection
3. **Simple retrieval**: Basic similarity search without re-ranking

### Production Improvements

1. **Semantic chunking**: Use LLM to generate Q&A pairs automatically
2. **Hybrid search**: Combine semantic + keyword search
3. **Re-ranking**: Use cross-encoder for better relevance
4. **Multi-tenancy**: Separate collections per customer/org
5. **Caching**: Cache frequent queries with Langfuse
6. **A/B testing**: Test different prompt versions

---

## Assignment Checklist

### Question 1 Requirements

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| `/chat` endpoint | ✅ | FastAPI with message + history |
| System prompt | ✅ | Langfuse prompt management |
| RAG pipeline | ✅ | Qdrant + Q&A pairs |
| Escalation flag | ✅ | Based on retrieval + confidence |
| Sources tracking | ✅ | Included in response |
| .env.example | ✅ | Provided |
| README with explanations | ✅ | Setup, chunking, embedding choices |

### README Sections

1. **Setup Instructions** - Installation and configuration
2. **Q1 Explanations**:
   - System prompt structure
   - Chunking strategy (Q&A pairs)
   - Embedding model choice (text-embedding-3-small)
   - One limitation + production improvement

---

## Next Steps

1. ✅ Design document approved
2. ⏳ Create implementation plan (writing-plans skill)
3. ⏳ Implement core services
4. ⏳ Implement API endpoints
5. ⏳ Write tests
6. ⏳ Create README
