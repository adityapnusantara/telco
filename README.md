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

### Create System Prompt in Langfuse

Create a chat prompt in Langfuse with name `telco-customer-service-agent`:

```yaml
type: chat
prompt:
  - role: system
    content: |
      You are a customer service agent for {{company_name}}, a telecommunications provider.

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

      Escalation contact: {{escalation_contact}}

      Tone: Professional, helpful, concise

variables:
  company_name:
    type: string
    default: "MyTelco"
  escalation_contact:
    type: string
    default: "call 123 or use the MyTelco app"

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

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are your service plans?",
    "conversation_history": []
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

The system prompt is structured with:
1. Clear role definition - Sets boundaries as a Telco customer service agent
2. Explicit capabilities - Lists what the agent can and cannot do
3. Structured response format - Defines the JSON schema for responses (reply, confidence_score, escalate)
4. Confidence guidelines - Provides clear criteria for confidence scoring
5. Escalation criteria - Defines when to escalate to prevent hallucination
6. Tone guidance - Ensures consistent customer experience
7. Variables - `company_name` and `escalation_contact` for flexibility

## Chunking Strategy

Instead of using `RecursiveCharacterTextSplitter`, we use **Q&A pairs** as our chunking strategy:

### Why Q&A Pairs?

- Better semantic matching: Customer queries are naturally question-based
- Self-contained chunks: Each Q&A pair is complete, no context needed
- No broken context: No issues with chunk boundaries
- Easier maintenance: Can be created and validated manually

## Embedding Model

**Selected**: `text-embedding-3-small` with 512 dimensions

**Reasoning**:
- Cost-effective for this use case
- Fast response times
- Sufficient quality for document-based Q&A
- Smaller vectors = faster similarity search in Qdrant

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
