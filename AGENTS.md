# Repository Guidelines

## Project Structure & Module Organization
Core application code lives in `app/`:
- `app/api/` for FastAPI routes and request/response models.
- `app/services/llm/` for agent, chat orchestration, and callbacks.
- `app/services/rag/` for knowledge base loading, ingestion, retriever, and vector store.
- `app/core/` for configuration loading.
- `app/prompts/` for Langfuse prompt integration.

Tests are in `tests/` (pytest), scripts in `scripts/` (for example `scripts/ingest_kb.py`), sample data in `data/kb/`, and design/implementation notes in `docs/`.

## Build, Test, and Development Commands
- `poetry install` installs runtime and dev dependencies.
- `cp .env.example .env` then populate API credentials.
- `poetry run python run.py` starts the local dev server.
- `poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000` runs FastAPI with reload.
- `poetry run python scripts/ingest_kb.py` ingests Q&A docs into Qdrant.
- `poetry run pytest` runs all tests.
- `poetry run pytest --cov=app --cov-report=html` generates coverage output.

## Coding Style & Naming Conventions
Use Python 3.11+ and Pydantic/FastAPI idioms. Follow existing style:
- 4-space indentation, type hints on public functions, clear docstrings only when behavior is non-obvious.
- `snake_case` for functions/variables/files, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants/env keys.
- Keep modules focused by layer (`api`, `services/llm`, `services/rag`, `core`), and prefer dependency injection over globals.

## Testing Guidelines
Framework: `pytest` with `pytest-asyncio` and `httpx` for API tests.
- Name files `tests/test_*.py` and test functions `test_*`.
- Keep unit tests deterministic; mark external-service tests with `@pytest.mark.integration`.
- Run a focused file during development (example: `poetry run pytest tests/test_api_routes.py -v`) before full suite.

## Commit & Pull Request Guidelines
Git history follows Conventional Commit-style prefixes: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore(release):`.
- Use imperative, scoped subjects (example: `fix: handle missing prompt config in chat service`).
- Keep commits small and logically grouped.
- PRs should include: concise summary, linked issue/ticket, test evidence (`poetry run pytest` output), and API examples/screenshots when endpoint behavior changes.
