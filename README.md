# Telco Customer Service AI Agent

AI-powered customer service agent for a telecommunications company, built with FastAPI, LangChain, OpenAI GPT-4o, Qdrant, and Langfuse.

## Features

- 🤖 RAG-powered chatbot using Q&A knowledge base
- 🔍 Semantic search with Qdrant Cloud vector store
- 📊 Prompt management and tracing with Langfuse
- 🚀 Built with LangChain v1.2.11 and `create_agent`
- 🔄 Automatic escalation to human agents when needed
- 📝 Structured responses with confidence scores

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry
- OpenAI API key
- Langfuse account
- Qdrant Cloud account

### Installation

```bash
# Install dependencies
poetry install

# Copy environment template
cp .env.example .env

# Edit .env with your credentials
```

### Knowledge Base Setup

```bash
# Ingest Q&A documents into Qdrant
poetry run python scripts/ingest_kb.py
```

### Sample Knowledge Base

The knowledge base contains Q&A pairs converted from the 3 sample documents:

**Document 1 — Billing Policy**
- Bills generated on the 1st of every month
- Late payment fee: IDR 50,000 after 14 days overdue
- Billing dispute: within 30 days of invoice date
- Auto-pay via MyTelco app

**Document 2 — Service Plans**
- Basic Plan: IDR 99,000/month — 10GB data, unlimited calls
- Pro Plan: IDR 199,000/month — 50GB data, unlimited calls, 5GB hotspot
- Unlimited Plan: IDR 299,000/month — Unlimited data, calls, 20GB hotspot
- Free streaming access on weekends

**Document 3 — Troubleshooting Guide**
- Slow internet: restart device, check signal, toggle airplane mode
- Call quality: check network congestion via MyTelco app
- Billing errors: submit ticket via app or call 123
- SIM replacement: available at authorized stores with valid ID

These are stored in `data/kb/` as `billing_qna.json`, `plans_qna.json`, and `troubleshooting_qna.json`.

### Create System Prompt in Langfuse

Create a chat prompt in Langfuse with name `telco-customer-service-agent`:

```yaml
type: chat
prompt:
  - role: system
    content: |
      You are a customer service agent for MyTelco, a telecommunications provider.

      Your capabilities:
      - Answer questions about billing, service plans, and troubleshooting
      - Use ONLY information from the retrieved knowledge base
      - If the information is not in the knowledge base, acknowledge it honestly

      Structured Response Format:
      You must respond with a structured output containing:
      - reply: Your natural language response to the customer
      - confidence_score: A float from 0.0 to 1.0 indicating how confident you are in your answer
      - escalate: A boolean flag - set to true if the customer needs human assistance

      Confidence Guidelines:
      - 0.9-1.0: Direct answer found in knowledge base with clear, relevant information
      - 0.7-0.9: Answer based on knowledge base but requires some interpretation
      - 0.5-0.7: Partial information available, answer may be incomplete
      - 0.0-0.5: Limited or no relevant information found

      Escalation Criteria:
      Set escalate=true when:
      - User question cannot be answered with available information
      - Request requires capabilities outside agent scope (account changes, refunds)
      - User explicitly asks for a human agent
      - Sensitive issues (legal, fraud, billing disputes)
      - Confidence score is below 0.5

      Escalation contact: call 123 or use the MyTelco app

      Tone: Professional, helpful, concise

labels:
  - production
```

### Run the Application

```bash
# Development server
poetry run python run.py

# Or using uvicorn directly
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Usage

### Chat Endpoint

**Basic request:**
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are your service plans?"
  }'
```

**With session tracking:**
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How much is the late fee?",
    "session_id": "user-123-session-456",
    "conversation_history": [
      {"role": "user", "content": "When are bills generated?"},
      {"role": "assistant", "content": "Bills are generated on the 1st of every month."}
    ]
  }'
```

**Response Format:**

```json
{
  "reply": "We offer several service plans...",
  "confidence_score": 0.95,
  "escalate": false,
  "sources": ["plans_qna.json"]
}
```

**Response Fields:**
- `reply`: The agent's natural language response
- `confidence_score`: Confidence level (0.0-1.0) indicating answer quality
- `escalate`: Boolean flag - true if human agent assistance is recommended
- `sources`: List of knowledge base files used for the answer

## System Prompt Design

The system prompt is structured to ensure reliable, hallucination-free responses:

### Why This Structure?

1. **Clear Role Definition** - Sets explicit boundaries as a Telco customer service agent, preventing the LLM from claiming capabilities it doesn't have.

2. **Explicit Capabilities** - Lists exactly what the agent can (billing, plans, troubleshooting) and cannot do (account changes, refunds) - reducing out-of-scope responses.

3. **Structured Response Format** - Defines JSON schema (reply, confidence_score, escalate) enabling programmatic escalation decisions based on LLM's own confidence assessment.

4. **Confidence Guidelines** - Provides specific numeric ranges (0.0-1.0) with clear criteria for each level. This prevents arbitrary confidence values and ensures consistency.

5. **Escalation Criteria** - Explicitly lists scenarios requiring human intervention (legal, fraud, billing disputes, explicit human request). This is critical for customer satisfaction and compliance.

6. **Tone Guidance** - Ensures consistent "Professional, helpful, concise" responses aligned with Telco brand standards.

## Chunking Strategy

### Approach: Q&A Pairs (not Character-Based Splitting)

Instead of using `RecursiveCharacterTextSplitter` with fixed chunk sizes and overlap, we use **pre-processed Q&A pairs** as our chunking unit.

### Why Not Traditional Chunking?

Traditional RAG uses character-based splitting with:
- Chunk size: 500-1000 characters
- Overlap: 50-200 characters

**Problems for Customer Service Q&A:**
- Arbitrary chunk boundaries break question-answer pairs
- Important context often lost at chunk boundaries
- Customer queries are question-based, not paragraph-based
- Retrieval returns incomplete halves of answers

### Why Q&A Pairs?

Each Q&A pair is a self-contained semantic unit:

```json
{
  "question": "When are bills generated?",
  "answer": "Bills are generated on the 1st of every month.",
  "source": "billing_policy.md",
  "category": "billing"
}
```

**Benefits:**
- **Better semantic matching** - Questions match directly to customer queries
- **No broken context** - Each chunk is complete and self-contained
- **Higher precision** - Retrieval returns relevant, complete answers
- **Easier maintenance** - Subject matter experts can validate individual Q&A pairs
- **Natural conversation flow** - Mirrors how customers actually ask questions

**Trade-off:** Requires manual Q&A creation vs automatic document splitting. For production, we could use LLM to generate Q&A pairs automatically from raw documents.

## Embedding Model Choice

**Selected:** OpenAI `text-embedding-3-small` with **1536 dimensions**

### Why This Model?

| Factor | Decision | Reasoning |
|--------|----------|-----------|
| **Cost** | Small model | ~10x cheaper than `text-embedding-3-large` - critical for production scale |
| **Speed** | Small model | Faster inference = lower latency for customer-facing API |
| **Quality** | Small model | Sufficient for document-based Q&A where answers are explicit in text |
| **Dimensions** | 1536 | Default dimension for `text-embedding-3-small` - good balance of quality and performance |

### Alternative Considered: `text-embedding-3-large`

- **Pros:** Slightly better semantic understanding, supports multilingual better
- **Cons:** 4x more expensive, slower, higher dimensional vectors (3072)
- **Decision:** Not justified for this use case - Q&A pairs have clear semantic relationships

### Dimensionality Reduction Option

The `text-embedding-3-small` model supports dimensionality reduction (can use 512 instead of 1536). For production:
- **Smaller dimensions (512)** = Faster search, lower storage costs
- **Trade-off:** Potential slight quality degradation
- **Current implementation:** Uses 1536 for best retrieval quality

## Limitations & Production Improvements

### Current Limitations

1. Manual Q&A creation: Documents are manually converted to Q&A pairs
2. Simple retrieval: Basic similarity search without re-ranking
3. Single collection: All documents in one Qdrant collection

### Production Improvements

1. Automatic Q&A generation: Use LLM to generate Q&A pairs from documents
2. Hybrid search: Combine semantic + keyword search
3. Re-ranking: Use cross-encoder for better relevance
4. Multi-tenancy: Separate collections per customer/organization

## Project Structure

```
telco/
├── app/
│   ├── main.py              # FastAPI application with startup/shutdown events
│   ├── api/
│   │   ├── models.py        # Pydantic request/response models
│   │   └── routes/
│   │       └── chat.py      # /chat endpoint with Depends injection
│   ├── services/
│   │   ├── llm/
│   │   │   ├── agent.py     # Agent class with LangChain create_agent
│   │   │   ├── callbacks.py # CallbackHandler class (Langfuse tracing)
│   │   │   └── chat.py      # ChatService class (orchestration)
│   │   └── rag/
│   │       ├── vector_store.py  # VectorStore class (Qdrant wrapper)
│   │       ├── retriever.py     # RetrieverTool class (search tool)
│   │       ├── models.py        # QNADocument Pydantic model
│   │       ├── knowledge_base.py
│   │       └── ingestion.py
│   ├── core/
│   │   └── config.py        # Configuration
│   └── prompts/
│       └── langfuse.py      # Langfuse integration
├── data/kb/                 # Knowledge base Q&A files
├── scripts/
│   └── ingest_kb.py        # Ingestion script
├── tests/                   # Test suite
├── .env.example
├── pyproject.toml
└── README.md
```

## Architecture

The application uses **class-based services with dependency injection**:

- **VectorStore** - Qdrant vector store wrapper with explicit initialization
- **CallbackHandler** - Langfuse tracing handler wrapper
- **RetrieverTool** - Knowledge base search tool with VectorStore dependency
- **Agent** - LangChain agent with structured output (StructuredChatResponse)
- **ChatService** - Chat orchestration, extracts and validates structured responses

**Lifecycle Management:**
- All services are initialized during FastAPI startup event
- Services are stored in `app.state` for application-wide access
- Routes use FastAPI's `Depends()` for dependency injection
- No global singleton functions

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API Framework | FastAPI 0.135.1 |
| Agent Framework | LangChain 1.2.11 |
| LLM | OpenAI GPT-4o |
| Vector Store | Qdrant Cloud |
| Embeddings | OpenAI text-embedding-3-small |
| Prompt Management | Langfuse 4.0.0 |
| Tracing | Langfuse CallbackHandler |

## License

MIT License - Built as a technical assignment for AI Engineer position.
