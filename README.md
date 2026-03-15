# Telco Customer Service AI Agent

AI-powered customer service agent for a telecommunications company, built with FastAPI, LangChain, OpenAI GPT-4o, Qdrant, and Langfuse.

> **Disclaimer:** This project setup assumes Conda is already installed on your device.

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
# Setup environment and install dependencies
sh scripts/init.sh

# Copy environment template
cp .env.example .env
```

After creating `.env`, fill in the required credentials:

- `OPENAI_API_KEY`
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY`

Optional but commonly set:
- `LANGFUSE_BASE_URL` (default: `https://cloud.langfuse.com`)
- `QDRANT_COLLECTION_NAME` (default: `telco_knowledge_base`)

### Knowledge Base Setup

```bash
# Ingest Q&A documents into Qdrant
poetry run python scripts/ingest_kb.py
```

### Ingestion Flow

```text
scripts/ingest_kb.py
        |
        v
Load markdown sources (data/kb_md/*.md)
        |
        v
Clear existing JSON artifacts (data/kb_json/*.json)
        |
        v
Extract Q&A per markdown source (extraction agent)
        |
        v
Write per-source JSON files (data/kb_json/*.json)
        |
        v
Load generated Q&A JSON documents
        |
        v
Convert each Q&A into LangChain Document
        |
        v
Add documents to Qdrant collection
        |
        v
Print summary counts (generated files + ingested docs)
```

This flow is executed by `run_full_ingestion()` and performs extraction plus vector ingestion in one command.

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

Markdown sources are stored in `data/kb_md/`.
Generated Q&A JSON artifacts are written to `data/kb_json/` during ingestion.

### Create Prompts in Langfuse

Create these prompts in Langfuse:
- `telco-customer-service-agent` (main chat agent system prompt)
- `telco-customer-service-classification-user` (classification system prompt)
- `telco-customer-service-classification-system` (classification user prompt template)
- `telco-kb-extraction-system` (extraction system prompt + model config)
- `telco-kb-extraction-user` (extraction user prompt template)

### Run the Application

```bash
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
  "sources": ["service_plans.md"]
}
```

**Response Fields:**
- `reply`: The agent's natural language response
- `confidence_score`: Confidence level (0.0-1.0) from the classification agent
- `escalate`: Boolean flag from the classification agent
- `sources`: List of knowledge base files used for the answer

### Streaming Endpoint (SSE)

Use SSE if you want token-by-token output over HTTP:

```bash
curl -N -X POST "http://localhost:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are your service plans?",
    "session_id": "user-123-session-456",
    "conversation_history": []
  }'
```

Example SSE events:

```text
data: {"type":"token","content":"We"}

data: {"type":"token","content":" offer"}

data: {"type":"end","reply":"We offer...","confidence_score":0.9,"escalate":false,"sources":["service_plans.json"]}
```

### WebSocket Endpoint

Use WebSocket for bidirectional streaming (send message/cancel while connected):

```bash
wscat -c ws://localhost:8000/chat/stream/ws
```

Send a message:

```json
{"type":"message","message":"Halo","session_id":"test123","conversation_history":[]}
```

Optional cancel event:

```json
{"type":"cancel"}
```

Example events from server:

```json
{"type":"token","content":"Hello"}
{"type":"token","content":"!"}
{"type":"end","reply":"Hello! How can I assist you today with your MyTelco services?","confidence_score":0.9,"escalate":false,"sources":null}
```

## Q1 "What to Explain" Mapping

This section explicitly maps to Question 1 requirements in the assignment:

1. **System prompt and rationale**  
See [System Prompt Design](#system-prompt-design).  
The prompt enforces role boundaries, retrieval-grounded answers, and escalation behavior.

2. **Chunking strategy and reasoning**  
See [Chunking Strategy](#chunking-strategy).  
We use Q&A-pair chunks (instead of fixed-size character chunks) because the use case is question-driven and benefits from complete question-answer semantic units.

3. **Embedding model choice and rationale**  
See [Embedding Model Choice](#embedding-model-choice).  
We selected `text-embedding-3-small` (1536 dims) as a pragmatic balance of quality, speed, and operational simplicity.

4. **One limitation and production improvement**  
See [Limitations & Production Improvements](#limitations--production-improvements).  
Current limitation: basic retrieval without reranking. Planned improvement: hybrid retrieval + reranking.

## System Prompt Design

The system prompt is structured to ensure reliable, hallucination-free responses:

### Why This Structure?

1. **Clear Role Definition** - Sets explicit boundaries as a Telco customer service agent, preventing the LLM from claiming capabilities it doesn't have.

2. **Explicit Capabilities** - Lists exactly what the agent can (billing, plans, troubleshooting) and cannot do (account changes, refunds) - reducing out-of-scope responses.

3. **Natural Reply Output** - Main agent returns natural text only, while metadata (`confidence_score`, `escalate`) is generated by a dedicated classification agent.

4. **Confidence Guidelines** - Provides specific numeric ranges (0.0-1.0) with clear criteria for each level. This prevents arbitrary confidence values and ensures consistency.

5. **Escalation Criteria** - Explicitly lists scenarios requiring human intervention (legal, fraud, billing disputes, explicit human request). This is critical for customer satisfaction and compliance.

6. **Tone Guidance** - Ensures consistent "Professional, helpful, concise" responses aligned with Telco brand standards.

### Prompt Name Mapping (Runtime Config)

This section documents the active prompts managed in Langfuse (verbatim).

- `AGENT_PROMPT_NAME` -> `telco-customer-service-agent`
- `CLASSIFICATION_SYSTEM_PROMPT_NAME` -> `telco-customer-service-classification-user`
- `CLASSIFICATION_USER_PROMPT_NAME` -> `telco-customer-service-classification-system`
- `EXTRACTION_SYSTEM_PROMPT_NAME` -> `telco-kb-extraction-system`
- `EXTRACTION_USER_PROMPT_NAME` -> `telco-kb-extraction-user`

#### 1) Agent Prompt

- Name: `telco-customer-service-agent`
- `model_config`: `{"model": "gpt-4o", "temperature": 0}`

```text
You are a customer service agent for MyTelco, a telecommunications provider.

Your capabilities:
- Answer questions about billing, service plans, and troubleshooting
- Use ONLY information from the retrieved knowledge base
- If the information is not in the knowledge base, acknowledge it honestly

Escalation rules:
- Escalate to human if you cannot confidently answer from the knowledge base
- DO NOT make up or hallucinate information
- Clearly state when you don't know something

Tone: Professional, helpful, concise

Escalation contact: call 123 or use the MyTelco app
```

#### 2) Classification System Prompt

- Name: `telco-customer-service-classification-user`
- `model_config`: `{"model": "gpt-4o", "temperature": 0}`

```text
You are a customer service response quality classifier. Analyze the reply and provide metadata.

Reply: {{reply}}

Context: {{context}}

Scoring Guidelines:

**confidence_score** (0.0-1.0):
- 0.9-1.0: Sources available, answer is specific and complete
- 0.7-0.8: Sources available but answer is somewhat vague or incomplete
- 0.5-0.6: No sources available OR answer is generic/could be wrong
- 0.3-0.4: Cannot answer, apologizing, or clearly uncertain
- 0.0-0.2: Completely unable to help

**escalate** (true/false):
Set to TRUE if:
- Customer explicitly asks for human agent
- Customer wants to do something outside agent's scope (account changes, refunds, cancellations)
- Sensitive topics: legal threats, fraud reports, billing disputes, formal complaints
- Question cannot be answered with available information
- Customer seems frustrated or explicitly dissatisfied
- No sources available AND answer expresses inability to help

Set to FALSE if:
- Sources available and answer directly addresses the question
- Generic but helpful information is provided
- Answer acknowledges limitations but provides useful guidance

Return only JSON: {"confidence_score": 0.8, "escalate": false}
```

#### 3) Classification User Prompt

- Name: `telco-customer-service-classification-system`
- `model_config`: `{"model": "gpt-4o", "temperature": 0}`

```text
You are a customer service response classifier.
Analyze replies and provide metadata about confidence and escalation needs.
Always respond with valid JSON matching the required schema.
```

#### 4) Extraction System Prompt

- Name: `telco-kb-extraction-system`
- `model_config`: `{"model": "gpt-4o", "temperature": 0}`

```text
You are a strict information extraction assistant. Extract factual Q&A pairs only from the provided markdown. Do not add external information. Keep answers concise.
```

#### 5) Extraction User Prompt

- Name: `telco-kb-extraction-user`
- `model_config`: `{"model": "gpt-4o", "temperature": 0}`

```text
Extract Q&A pairs from this markdown content.
Source filename: {{source}}
Category: {{category}}
Markdown content:
{{markdown_content}}
```

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

| Factor | Reasoning |
|--------|-----------|
| **Simplicity** | Managed API - zero infrastructure to set up or maintain |
| **Speed** | Fast inference via optimized API (~50-100ms latency) |
| **Quality** | Strong performance on English Q&A where answers are explicit in text |
| **Dimensions** | 1536 - good balance of quality and performance |
| **Cost-effective for this use case** | Q&A pairs are clear and don't require complex retrieval |

### Alternative: BAAI/bge-m3

**BAAI/bge-m3** is a powerful open-source embedding model worth considering:

- **Dimensions:** 1024 (33% less storage per vector)
- **Multilingual:** Optimized for 100+ languages
- **Long documents:** Handles up to 8192 tokens
- **Cost:** Free to use (self-hosted)

**Why not chosen for this project?**

The main consideration is **infrastructure complexity**:

| Requirement | text-embedding-3-small | BAAI/bge-m3 |
|-------------|------------------------|-------------|
| Deployment | API call (zero setup) | Self-hosted (GPU/CPU server) |
| Maintenance | Managed by OpenAI | You maintain the server |
| Scaling | Automatic | Manual scaling |
| Cost | Pay per token | Infrastructure + compute costs |

For this Telco customer service Q&A system:
- **Quick deployment** → managed API is the simplest path
- **Data volume is manageable** → current ingestion size makes text-embedding-3-small cost-effective

**Recommendation:** If data ingestion grows significantly (hundreds of thousands of Q&A pairs), consider migrating to BAAI/bge-m3 for better cost efficiency.

**When to consider migrating to BAAI/bge-m3?**

1. **Cost optimization at scale** - Self-hosting becomes cheaper at high query volumes
2. **Multilingual expansion** - Need for 100+ language support
3. **Data privacy** - Requirement to keep all processing in-house
4. **Long document handling** - Frequently need to search through >4000 token documents

### Dimensionality Option

The `text-embedding-3-small` model supports dimensionality reduction (512 instead of 1536):
- **Benefit:** 67% less storage, faster search
- **Trade-off:** Slight quality degradation
- **Current:** Uses 1536 for best retrieval quality

For production, test both dimensions with your specific Q&A dataset to find the right balance.

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
├── data/kb_md/              # Markdown knowledge base sources
├── data/kb_json/            # Generated Q&A JSON artifacts
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
- **Agent** - LangChain agent that returns natural text replies
- **ChatService** - Chat orchestration plus metadata classification (`confidence_score`, `escalate`)

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
