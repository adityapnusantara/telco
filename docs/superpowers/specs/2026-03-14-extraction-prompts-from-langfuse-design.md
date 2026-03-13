# Extraction Prompts From Langfuse Design

Date: 2026-03-14
Status: Drafted and user-approved (design stage)

## Summary
Move extraction prompting in `app/services/rag/ingestion.py` from hardcoded strings to Langfuse-managed prompts, following the same pattern used by `Agent` and `ChatService`.

The extraction flow will use two prompt objects from Langfuse:
- system prompt (raw string)
- user prompt template (compiled at runtime with extraction context)

Model selection for extraction will come from Langfuse prompt config (primary source).

## Goals
- Remove hardcoded `EXTRACTION_SYSTEM_PROMPT` and inline user extraction prompt from ingestion logic.
- Manage extraction behavior in Langfuse prompt management.
- Keep ingestion public interfaces and extraction output behavior unchanged.
- Keep model and temperature configuration for extraction centrally controlled via Langfuse.

## Non-Goals
- Refactoring all prompt-loading code into a new generic framework.
- Changing extraction output schema (`QNAExtractionResult`, `QNADocument`).
- Changing ingestion API shape (`extract_qna_from_markdown`, `run_full_ingestion`).

## Proposed Changes

### 1. Configuration
Update `app/core/config.py` with two new prompt-name configs:
- `EXTRACTION_SYSTEM_PROMPT_NAME` (default: `telco-kb-extraction-system`)
- `EXTRACTION_USER_PROMPT_NAME` (default: `telco-kb-extraction-user`)

`EXTRACTION_MODEL` can remain in config for backward compatibility or transitional fallback, but extraction model source of truth becomes Langfuse `model_config`.

### 2. Langfuse Prompt Loader
Add a new function in `app/prompts/langfuse.py`:
- `get_extraction_prompt()` returning:
  - `system_prompt`: `str`
  - `user_prompt`: Langfuse prompt object (for `.compile(...)`)
  - `model_config`: `dict` (model, temperature)

Behavior mirrors existing `get_classification_prompt()` style.

### 3. Ingestion Prompt Wiring
Update `app/services/rag/ingestion.py`:
- Remove module-level hardcoded `EXTRACTION_SYSTEM_PROMPT`.
- In `_build_extraction_agent()`:
  - call `get_extraction_prompt()`
  - instantiate `ChatOpenAI` with model/temperature from `model_config`
  - create agent with Langfuse `system_prompt`
- In `_extract_qna_for_source(...)`:
  - compile user prompt using
    - `source`
    - `category`
    - `markdown_content`
  - send compiled prompt as user message to extraction agent

No change to normalization behavior that overwrites extracted `source` and `category` to match file metadata.

## Data Flow
1. `extract_qna_from_markdown()` builds extraction agent.
2. `_build_extraction_agent()` fetches extraction prompt bundle from Langfuse.
3. For each markdown source, `_extract_qna_for_source()` compiles user prompt with source context.
4. Agent returns `structured_response` as `QNAExtractionResult`.
5. Result is normalized and written into JSON artifacts as before.

## Error Handling
- If Langfuse prompt fetch fails or returns malformed data, extraction should fail fast.
- If `structured_response` is missing, existing `ValueError` behavior remains.
- If prompt compilation fails due to missing variables, fail fast to surface prompt/template mismatch early.

## Testing Strategy

### Unit Tests
1. `tests/test_langfuse.py`
- Add test for `get_extraction_prompt()`:
  - verifies system prompt fetch by configured name
  - verifies user prompt fetch by configured name
  - verifies returned `model_config`

2. `tests/test_ingestion.py`
- Update extraction tests to mock Langfuse extraction prompt provider.
- Assert user prompt `.compile(...)` called with expected variables (`source`, `category`, `markdown_content`).
- Keep assertions for overwrite behavior and orchestration order.

### Regression Coverage
- Ensure no behavior changes in:
  - generated JSON file naming
  - extracted item count
  - source/category normalization
  - full pipeline orchestration (`run_full_ingestion`).

## Rollout Notes
- Create prompts in Langfuse:
  - `telco-kb-extraction-system`
  - `telco-kb-extraction-user`
- Ensure system prompt config includes `model` and `temperature`.
- Deploy env with new prompt-name variables only if names differ from defaults.

## Risks and Mitigations
- Risk: Prompt variables in Langfuse template mismatch code variables.
  - Mitigation: Unit test compile call args and fail fast on compile errors.
- Risk: Missing/incorrect model config on Langfuse prompt.
  - Mitigation: Validate config shape in tests; optionally apply fallback defaults during implementation.

## Success Criteria
- Ingestion no longer contains hardcoded extraction prompt text.
- Extraction model/temperature are sourced from Langfuse config.
- Existing ingestion tests pass with updated prompt-loading mocks.
- No changes required for ingestion endpoint/script usage.
