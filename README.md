# Telco Customer Service AI Agent

AI-powered customer service agent for a telecommunications company, built with FastAPI, LangChain, OpenAI GPT-4o, Qdrant, and Langfuse.

## What This Project Does

- Exposes a `/chat` API endpoint for telco customer support questions.
- Uses RAG over a Telco knowledge base (billing, plans, troubleshooting).
- Returns natural-language replies plus metadata: `confidence_score` and `escalate`.
- Applies a hard safety rule: if retrieval sources are missing, the response is escalated to a human.

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry
- OpenAI API key
- Langfuse account
- Qdrant Cloud account

### Install

```bash
poetry install
cp .env.example .env
```

Set credentials in `.env`.

Optional prompt-name overrides (defaults shown):

```bash
AGENT_PROMPT_NAME=telco-customer-service-agent
CLASSIFICATION_SYSTEM_PROMPT_NAME=telco-customer-service-classification-user
CLASSIFICATION_USER_PROMPT_NAME=telco-customer-service-classification-system
EXTRACTION_SYSTEM_PROMPT_NAME=telco-kb-extraction-system
EXTRACTION_USER_PROMPT_NAME=telco-kb-extraction-user
```

## Run the Application

```bash
poetry run python run.py
# or
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API base URL: `http://localhost:8000`

## Ingest the Knowledge Base

```bash
poetry run python scripts/ingest_kb.py
```

Ingestion flow:
- Markdown sources from `data/kb_md/`
- Extracted Q&A JSON written to `data/kb_json/`
- Q&A documents embedded and stored in Qdrant

## Use the API

### Request

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are your service plans?",
    "conversation_history": []
  }'
```

### Response

```json
{
  "reply": "We offer several service plans...",
  "confidence_score": 0.95,
  "escalate": false,
  "sources": ["service_plans.md"]
}
```

Response fields:
- `reply`: natural-language answer from main agent
- `confidence_score`: score from classification agent
- `escalate`: escalation flag from classification agent or hard rule
- `sources`: retrieval sources used from the knowledge base

## Configure Langfuse Prompts

Use these prompt names:
- `telco-customer-service-agent`
- `telco-customer-service-classification-system`
- `telco-customer-service-classification-user`
- `telco-kb-extraction-system`
- `telco-kb-extraction-user`

Canonical prompt content:

### `telco-customer-service-agent`

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

### `telco-customer-service-classification-system`

```text
You are a customer service response classifier.
Analyze replies and provide metadata about confidence and escalation needs.
Always respond with valid JSON matching the required schema.
```

### `telco-customer-service-classification-user`

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

### `telco-kb-extraction-system`

```text
You are a strict information extraction assistant. Extract factual Q&A pairs only from the provided markdown. Do not add external information. Keep answers concise.
```

### `telco-kb-extraction-user`

```text
Extract Q&A pairs from this markdown content.
Source filename: {{source}}
Category: {{category}}
Markdown content:
{{markdown_content}}
```

## How It Works (High-Level)

- `main agent` retrieves from KB and generates natural-language answer.
- `classification agent` assigns `confidence_score` and `escalate`.
- Hard rule: if retrieval has no sources, system returns fallback and `escalate=true`.

## Q1 "What to Explain" Mapping

1. System prompt and rationale:
See `Configure Langfuse Prompts` and `Deep Dive`.

2. Chunking strategy and reasoning:
See `Deep Dive -> Chunking Strategy`.

3. Embedding model choice and rationale:
See `Deep Dive -> Embedding Model Choice`.

4. One limitation and production improvement:
See `Deep Dive -> Limitations & Production Improvements`.

## Deep Dive

### System Prompt Design

- Clear role and capability boundaries reduce out-of-scope behavior.
- Retrieval-grounded answering reduces hallucination risk.
- Explicit escalation instructions improve failure handling.

### Chunking Strategy

Approach: Q&A-pair chunking (not character splitting).

Why:
- Customer queries are question-shaped.
- Q&A pairs preserve complete semantic units.
- Better precision than arbitrary chunk boundaries for this use case.

Trade-off:
- Requires controlled extraction/curation step.

### Embedding Model Choice

Selected: OpenAI `text-embedding-3-small` (1536 dimensions).

Why:
- Good quality/latency balance.
- Managed API reduces infra complexity.
- Fits current dataset scale.

### Limitations & Production Improvements

Current limitations:
- Basic similarity retrieval without re-ranking.
- Single Qdrant collection.

Production improvements:
- Hybrid retrieval (semantic + keyword).
- Re-ranking layer.
- Better multi-tenant partitioning strategy.

## Project Structure

```text
telco/
├── app/
│   ├── api/
│   │   ├── models.py
│   │   └── routes/chat.py
│   ├── core/config.py
│   ├── prompts/langfuse.py
│   ├── services/
│   │   ├── llm/
│   │   │   ├── agent.py
│   │   │   ├── callbacks.py
│   │   │   └── chat.py
│   │   └── rag/
│   │       ├── ingestion.py
│   │       ├── knowledge_base.py
│   │       ├── models.py
│   │       ├── retriever.py
│   │       └── vector_store.py
│   └── main.py
├── data/
│   ├── kb_md/
│   └── kb_json/
├── scripts/ingest_kb.py
├── tests/
├── .env.example
├── pyproject.toml
└── README.md
```

## Tech Stack

- FastAPI
- LangChain (`create_agent`)
- OpenAI GPT-4o + `text-embedding-3-small`
- Qdrant Cloud
- Langfuse (prompt management + tracing)

## License

MIT License
