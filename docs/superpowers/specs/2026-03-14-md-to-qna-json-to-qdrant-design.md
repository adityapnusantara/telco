# MD-to-QnA Extraction and Ingestion Design

## Objective
Replace the current JSON-first ingestion flow with a Markdown-first pipeline:

1. Load 3 source knowledge-base documents from `.md` files (one file per document).
2. Extract QnA pairs from each `.md` using `create_agent` with structured output.
3. Save extracted QnA to `.json` files using overwrite behavior.
4. Embed and store extracted QnA into Qdrant.

This design only changes ingestion; runtime chat/retrieval behavior remains unchanged.

## Scope
In scope:
- New source directory for markdown knowledge base files.
- Extraction stage using LLM agent and schema-validated output.
- JSON artifact generation (`overwrite` mode).
- Embedding stage from generated JSON into Qdrant.
- Tests for extraction, overwrite behavior, and end-to-end orchestration.

Out of scope:
- Changes to `/chat` API contract.
- Changes to retrieval logic used at runtime.
- Multi-file incremental hashing or versioned artifact strategy.

## Source and Artifact Layout
Source markdown (source of truth):
- `data/kb_md/billing_policy.md`
- `data/kb_md/service_plans.md`
- `data/kb_md/troubleshooting_guide.md`

Generated JSON artifacts:
- `data/kb_json/billing_policy.json`
- `data/kb_json/service_plans.json`
- `data/kb_json/troubleshooting_guide.json`

## Architecture
Pipeline is split into two deterministic stages:

1. Extraction stage:
- Input: `data/kb_md/*.md`
- Process: per-file LLM extraction via `create_agent` with structured schema
- Output: overwrite `data/kb_json/<stem>.json`

2. Embedding stage:
- Input: generated `data/kb_json/*.json`
- Process: convert QnA records to LangChain `Document`
- Output: `add_documents` into Qdrant collection

Single entrypoint remains `scripts/ingest_kb.py`, which orchestrates both stages.

## Data Model
Canonical QnA schema remains unchanged for compatibility:

- `question: str`
- `answer: str`
- `source: str`
- `category: str`

`source` is normalized to originating markdown filename.
`category` is derived from filename slug (or top-level heading fallback).

## Detailed Data Flow
### Step 1: Discover markdown files
- Scan `data/kb_md/*.md`.
- If zero files found, fail fast with explicit error.

### Step 2: Extract QnA per markdown file
- Read full markdown content.
- Call extraction agent (`create_agent`) with:
  - system prompt for faithful extraction only from provided text
  - structured response format returning list of QnA items
- Validate each item via `QNADocument`.
- Normalize metadata fields (`source`, `category`).
- Write JSON output in overwrite mode.

### Step 3: Ingest generated JSON into vector store
- Load all JSON files from `data/kb_json`.
- Convert each QnA to `Document`:
  - `page_content = "Question: ...\nAnswer: ..."`
  - metadata includes `source`, `category`, `question`
- Upsert documents using existing `VectorStore` integration.

### Step 4: Summary output
- Log per-file extraction counts.
- Log total markdown files processed and total QnA ingested.

## Error Handling
Fail-fast policy for consistency:
- No markdown files found: stop run.
- LLM extraction failure for any file: stop run.
- Schema validation failure for extracted item(s): stop run and report file/index.
- JSON write failure: stop run.
- Qdrant embedding/upsert failure: stop run.

Rationale: avoid partially refreshed datasets in CI/job execution.

## Module-Level Changes
### `app/services/rag/models.py`
- Add extraction output schema (e.g., wrapper model with list of QnA items) for structured LLM response.
- Keep existing `QNADocument` as canonical record model.

### `app/services/rag/knowledge_base.py`
- Add markdown source loader.
- Add JSON save helper (overwrite mode).
- Add loader for generated JSON directory.
- Keep legacy loader only if needed for backward compatibility.

### `app/services/rag/ingestion.py`
- Refactor into explicit orchestration functions:
  - `extract_qna_from_markdown(kb_md_dir, kb_json_dir) -> int`
  - `ingest_qna_json_directory(kb_json_dir) -> int`
  - `run_full_ingestion(kb_md_dir, kb_json_dir) -> tuple[int, int]`

### `scripts/ingest_kb.py`
- Call `run_full_ingestion("data/kb_md", "data/kb_json")`.
- Print extraction and embedding summaries.

## Testing Strategy
Unit tests (mock LLM and vector store):
- extraction creates one JSON output per markdown input.
- overwrite behavior replaces existing JSON content.
- extraction fails on invalid structured output.
- full ingestion orchestrates extraction then embedding in correct order.
- embedding stage reads generated JSON and builds expected metadata/content.

Integration behavior:
- existing integration tests can continue to validate runtime query behavior.
- ingestion-specific integration test is optional and should be marked integration.

## Operational Notes
- This flow requires valid LLM credentials during extraction stage.
- In local/dev without LLM credentials, extraction stage should fail explicitly.
- Existing Qdrant collection behavior is unchanged.

## Migration Plan
1. Add `data/kb_md` with 3 markdown source docs.
2. Introduce extraction + ingestion refactor.
3. Update `scripts/ingest_kb.py` to orchestrate both stages.
4. Add/adjust tests.
5. Update README ingestion section to document new source-of-truth and overwrite behavior.

## Acceptance Criteria
- Ingestion entrypoint consumes markdown files, not hand-authored QnA JSON.
- Generated JSON files are overwritten every run.
- QnA records extracted via `create_agent` and validated before persistence.
- Generated QnA records are embedded into Qdrant successfully.
- Tests cover extraction and orchestration paths with deterministic mocks.
