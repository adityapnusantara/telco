# AGENTS.md

This file provides guidance to Codex when working with code in this repository.

## Project Overview

Telco Customer Service AI Agent - A RAG-powered customer service chatbot for a telecommunications company, built with FastAPI, LangChain v1.2.11, OpenAI GPT-4o, Qdrant Cloud, and Langfuse for observability.

## Development Commands

### Environment Setup
```bash
# Install dependencies
poetry install

# Set up environment variables
cp .env.example .env
# Edit .env with your OPENAI_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, QDRANT_URL, QDRANT_API_KEY
```

### Running the Application
```bash
# Development server with hot reload
poetry run python run.py

# Alternative with uvicorn
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Knowledge Base Management
```bash
# Ingest Q&A documents into Qdrant vector store
poetry run python scripts/ingest_kb.py
```

### Testing
```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/test_api_models.py -v

# Run specific test
poetry run pytest tests/test_api_models.py::test_chat_request_valid -v

# Run with coverage
poetry run pytest --cov=app --cov-report=html

# Run tests and stop on first failure
poetry run pytest -x
```

## Architecture Overview

### RAG Pipeline Flow

```
User Request → FastAPI /chat → ChatService (via Depends)
    ↓
Agent class (invoke method) → Natural text reply
    ↓
RetrieverTool → VectorStore → Qdrant (semantic search)
    ↓
Langfuse CallbackHandler (tracing)
    ↓
Classification Agent → Metadata (confidence_score, escalate)
    ↓
Structured response (reply, confidence_score, escalate, sources)
```

### Service Architecture

The application uses **class-based services with explicit dependency injection**:

**LLM Services (`app/services/llm/`)**
- `agent.py` - `Agent` class for main RAG agent (natural text output)
- `callbacks.py` - `CallbackHandler` class wrapping Langfuse tracing
- `chat.py` - `ChatService` class for orchestration, includes:
  - Main RAG agent for generating replies
  - Classification agent for extracting metadata (confidence_score, escalate)
  - `ReplyClassification` model for structured classification output

**RAG Services (`app/services/rag/`)**
- `vector_store.py` - `VectorStore` class wrapping Qdrant client and embeddings
- `retriever.py` - `RetrieverTool` class wrapping LangChain @tool for knowledge base search
- `models.py` - `QNADocument` Pydantic model (question, answer, source, category)
- `knowledge_base.py` - Loads Q&A documents from JSON files in `data/kb/`
- `ingestion.py` - Converts Q&A pairs to LangChain Documents, stores in vector store

**Configuration & Integration**
- `app/core/config.py` - Environment variables via dotenv, all service credentials
- `app/prompts/langfuse.py` - Fetches prompts and configs from Langfuse Prompt Management:
  - `get_system_prompt()` - Main RAG agent system prompt
  - `get_model_config()` - Main agent model configuration
  - `get_classification_prompt_obj()` - Classification agent prompt template (with variables)
  - `get_classification_config()` - Classification agent model configuration
- `app/api/models.py` - Pydantic request/response models for `/chat` endpoint

### Lifecycle Management

**Startup Event (`app/main.py`)**
```python
@app.on_event("startup")
async def startup_event():
    # Services created in dependency order
    app.state.vector_store = VectorStore(...)
    app.state.callback_handler = CallbackHandler()
    retriever_tool = RetrieverTool(vector_store=app.state.vector_store)
    app.state.agent = Agent(vector_store=app.state.vector_store, retriever_tool=retriever_tool.tool)
    app.state.chat_service = ChatService(agent=app.state.agent, handler=app.state.callback_handler)
```

**Route Dependency Injection (`app/api/routes/chat.py`)**
```python
def get_chat_service(request: Request) -> ChatService:
    service = request.app.state.chat_service
    if service is None:
        raise HTTPException(status_code=503, detail="Services not initialized")
    return service

@router.post("/chat")
async def create_chat(request: ChatRequest, service: ChatService = Depends(get_chat_service)):
    return service.chat(...)
```

### LangChain Integration

Uses `create_agent` from `langchain.agents` (LangChain v1.2.11):
- Model: `ChatOpenAI(model="gpt-4o", temperature=0)`
- Tools: Single `search_knowledge_base` tool (from RetrieverTool)
- System prompt: Fetched from Langfuse, compiled with variables

### Knowledge Base Format

Q&A pairs stored as JSON in `data/kb/`:
- `billing_qna.json` - Billing policies, fees, disputes
- `plans_qna.json` - Service plans, pricing
- `troubleshooting_qna.json` - Technical support, SIM replacement

Each document has: `question`, `answer`, `source`, `category`

### Structured Output

The application uses structured output in two places:

**1. Classification Agent (ChatService)**
- Uses `create_agent` with `response_format=ReplyClassification`
- Returns `confidence_score` (0.0-1.0) and `escalate` (boolean)
- Uses gpt-4o model from Langfuse config
- Prompt template managed in Langfuse (`telco-customer-service-classification`)

**ReplyClassification Model:**
- `confidence_score`: Float (0.0-1.0) indicating answer confidence
- `escalate`: Boolean flag for human escalation

**2. Main RAG Agent**
- Outputs natural text (no structured format)
- Reply text is then classified for metadata

### Escalation Logic

Escalation is determined by the classification agent analyzing the reply and context:

- The classification agent uses a dedicated prompt template from Langfuse (`telco-customer-service-classification`)
- The `ReplyClassification` model returns both `confidence_score` and `escalate`
- Escalation criteria include:
  - User asks for human agent
  - Question cannot be answered with available information
  - Request requires capabilities outside agent scope (account changes, refunds)
  - Sensitive issues (legal, fraud, billing disputes)

**Classification Prompt Template (create in Langfuse):**
```
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

**Config for classification prompt:**
```json
{
    "model": "gpt-4o",
    "temperature": 0
}
```

### External Dependencies

- **Langfuse**: Used for both prompt management (system prompt) and tracing (CallbackHandler)
- **Qdrant Cloud**: Vector store with collection `telco_knowledge_base`
- **OpenAI**: GPT-4o for LLM, text-embedding-3-small for embeddings

### Important Implementation Notes

- **No global singletons** - All services are class-based with explicit dependency injection
- **app.state lifecycle** - Services created at startup, stored in app.state, injected via Depends()
- **Q&A pairs as chunks** - Each pair is self-contained, no text splitting needed
- **System prompt**: Hardcoded values in Langfuse (company_name="MyTelco", escalation_contact="call 123 or use the MyTelco app")
- **Tracing**: All requests automatically logged to Langfuse via CallbackHandler

### API Endpoints

- `POST /chat` - Main chat endpoint (async), takes `message`, optional `session_id`, optional `conversation_history`
- `POST /chat/stream` - SSE streaming endpoint for real-time token responses
- `WebSocket /chat/stream/ws` - WebSocket endpoint for bidirectional streaming
- `GET /` - Root endpoint with app info
- `GET /health` - Health check endpoint

### Streaming Implementation

Both `/chat/stream` (SSE) and `/chat/stream/ws` (WebSocket) stream **AI reply tokens only**, filtering out ToolMessage (knowledge base retrieval results).

**How to filter:**
```python
# ToolMessage has 'tool_call_id' attribute, AIMessage doesn't
if not hasattr(message_chunk, 'tool_call_id'):
    # Stream this as AI reply token
```

**SSE Response Format:**
```
data: {"type": "token", "content": "Hello"}
data: {"type": "token", "content": "!"}
data: {"type": "end", "reply": "Hello!", "confidence_score": 0.85, "escalate": false, "sources": [...]}
```

**Important:** The `chat()` method is async - route handlers must use `await service.chat(...)`.

### OpenTelemetry Warning Suppression

FastAPI async + OpenTelemetry produces "Failed to detach context" warnings. A logging filter is added in `app/main.py`:
```python
class OpenTelemetryContextFilter(logging.Filter):
    def filter(self, record):
        if "Failed to detach context" in record.getMessage():
            return False
        return True

otel_logger = logging.getLogger("opentelemetry.context")
otel_logger.addFilter(OpenTelemetryContextFilter())
```

### File Naming Conventions

- `app/api/routes/*.py` - FastAPI route handlers
- `app/services/{layer}/*.py` - Business logic services (llm/, rag/)
- `tests/test_*.py` - Test files mirroring app structure
