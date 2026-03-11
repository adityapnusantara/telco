# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
User Request → FastAPI /chat → ChatService
    ↓
LangChain Agent (create_agent)
    ↓
Retriever Tool → Qdrant Vector Store (semantic search)
    ↓
Langfuse CallbackHandler (tracing)
    ↓
Response with escalate flag based on keywords
```

### Service Architecture

The application is organized into three main service layers:

**LLM Services (`app/services/llm/`)**
- `agent.py` - LangChain `create_agent` with GPT-4o, wraps retriever tool
- `callbacks.py` - Langfuse `CallbackHandler` singleton for tracing
- `chat.py` - Chat orchestration, escalation detection via keyword matching

**RAG Services (`app/services/rag/`)**
- `models.py` - `QNADocument` Pydantic model (question, answer, source, category)
- `knowledge_base.py` - Loads Q&A documents from JSON files in `data/kb/`
- `vector_store.py` - Qdrant Cloud client, `text-embedding-3-small` (512 dims)
- `retriever.py` - LangChain `@tool` wrapped retriever for knowledge base search
- `ingestion.py` - Converts Q&A pairs to LangChain Documents, stores in vector store

**Configuration & Integration**
- `app/core/config.py` - Environment variables via dotenv, all service credentials
- `app/prompts/langfuse.py` - Fetches system prompt from Langfuse Prompt Management
- `app/api/models.py` - Pydantic request/response models for `/chat` endpoint

### LangChain Integration

Uses `create_agent` from `langchain.agents` (LangChain v1.2.11):
- Model: `ChatOpenAI(model="gpt-4o", temperature=0)`
- Tools: Single `search_knowledge_base` tool (retriever)
- System prompt: Fetched from Langfuse, compiled with variables

### Knowledge Base Format

Q&A pairs stored as JSON in `data/kb/`:
- `billing_qna.json` - Billing policies, fees, disputes
- `plans_qna.json` - Service plans, pricing
- `troubleshooting_qna.json` - Technical support, SIM replacement

Each document has: `question`, `answer`, `source`, `category`

### Escalation Logic

The `_should_escalate()` function checks for these keywords in responses:
- "cannot help", "don't know", "unable to assist"
- "speak to a human", "transfer to agent", "escalate"

When triggered, sets `escalate: true` in ChatResponse.

### External Dependencies

- **Langfuse**: Used for both prompt management (system prompt) and tracing (CallbackHandler)
- **Qdrant Cloud**: Vector store with collection `telco_knowledge_base`
- **OpenAI**: GPT-4o for LLM, text-embedding-3-small for embeddings

### Important Implementation Notes

- Singleton pattern used for agent, vector store, and callback handler
- Q&A pairs used as chunks instead of text splitting - each is self-contained
- System prompt variables: `company_name` (default: "MyTelco"), `escalation_contact` (default: "call 123 or use the MyTelco app")
- All tracing automatically logged to Langfuse via CallbackHandler passed to agent.invoke()

### API Endpoints

- `POST /chat` - Main chat endpoint, takes `message`, optional `conversation_id`, optional `conversation_history`
- `GET /` - Root endpoint with app info
- `GET /health` - Health check endpoint

### File Naming Conventions

- `app/api/routes/*.py` - FastAPI route handlers
- `app/services/{layer}/*.py` - Business logic services (llm/, rag/)
- `tests/test_*.py` - Test files mirroring app structure
