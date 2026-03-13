# README Task-Oriented Cleanup Design

Date: 2026-03-14
Status: Drafted and user-approved (design stage)

## Summary
Refactor `README.md` into a task-oriented structure while preserving technical depth needed for assignment review.

The README will prioritize fast onboarding and reviewer scanning first, then move detailed rationale into a deep-dive section.

## Goals
- Make README easier to follow in execution order (setup -> run -> use -> understand).
- Keep assignment alignment explicit (Question 1 "What to Explain").
- Eliminate redundancy and inconsistent terminology.
- Lock prompt examples to user-approved source-of-truth prompt text.

## Non-Goals
- No code changes.
- No architecture or feature changes.
- No additional documentation files unless required by scope creep.

## Source-of-Truth Prompts
These exact prompts are the canonical values for README examples:

1. `telco-customer-service-agent`
```
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

2. `telco-customer-service-classification-system`
```
You are a customer service response classifier.
Analyze replies and provide metadata about confidence and escalation needs.
Always respond with valid JSON matching the required schema.
```

3. `telco-customer-service-classification-user`
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

4. `telco-kb-extraction-system`
```
You are a strict information extraction assistant. Extract factual Q&A pairs only from the provided markdown. Do not add external information. Keep answers concise.
```

5. `telco-kb-extraction-user`
```
Extract Q&A pairs from this markdown content.
Source filename: {{source}}
Category: {{category}}
Markdown content:
{{markdown_content}}
```

## Proposed README Structure (Task-Oriented)

1. What This Project Does
2. Quick Start
3. Run the Application
4. Ingest the Knowledge Base
5. Use the API
6. Configure Langfuse Prompts
7. How It Works (High-Level)
8. Q1 "What to Explain" Mapping
9. Deep Dive
10. Project Structure
11. Tech Stack

## Content Cleanup Rules
- Keep examples runnable and minimal (one primary curl + one sample response).
- Remove repeated explanations across sections.
- Keep terminology consistent:
  - `main agent`: natural-language reply generator
  - `classification agent`: `confidence_score` and `escalate` metadata
  - `retrieval sources`: KB-backed context used by the main agent
- Ensure statements match current behavior:
  - no-source retrieval path forces escalation
  - ingestion flow is `data/kb_md` -> generated `data/kb_json` -> vector store

## Data/Behavior Consistency Checks
- Verify paths and commands:
  - `poetry run python scripts/ingest_kb.py`
  - `poetry run python run.py`
- Verify prompt names and mapping in README match config defaults.
- Verify API response examples align with current response schema.

## Testing and Validation Plan
- Manual validation checklist after editing README:
  - Heading order is task-oriented.
  - Q1 mapping is explicit and intact.
  - Prompt snippets match source-of-truth values above.
  - No contradictory claims vs current implementation.

## Risks and Mitigations
- Risk: README gets shorter but loses decision rationale.
  - Mitigation: move detail to `Deep Dive`, not delete.
- Risk: prompt examples diverge from actual Langfuse setup.
  - Mitigation: keep source-of-truth prompt block in one section and reference it.

## Success Criteria
- A reviewer can run the project and test `/chat` quickly without reading deep sections first.
- Q1 rubric items are immediately discoverable.
- Prompt documentation is accurate to current setup and recent prompt changes.
