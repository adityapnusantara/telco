#!/usr/bin/env python
"""Run Langfuse SDK experiment on an existing Langfuse dataset.

This script executes the current Telco RAG agent against dataset items hosted in
Langfuse. Scores are expected to be configured and computed in Langfuse UI evaluators.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage
from langfuse import get_client

from app.core.config import config
from app.services.llm.agent import Agent
from app.services.rag.retriever import RetrieverTool
from app.services.rag.vector_store import VectorStore


def _to_text(value: Any) -> str:
    """Convert dataset input/expected_output values to a stable text form."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        # Common conversational dataset shape
        if isinstance(value.get("message"), str):
            return value["message"]
        if isinstance(value.get("input"), str):
            return value["input"]
    return json.dumps(value, ensure_ascii=False)


def _extract_reply(result: Any) -> str:
    """Extract final AI reply from LangChain agent invoke result."""
    messages = result.get("messages", []) if isinstance(result, dict) else []
    for message in reversed(messages):
        # ToolMessage usually has tool_call_id; skip those.
        if hasattr(message, "tool_call_id"):
            continue

        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            # Handle rich content arrays from some model providers.
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
            if text_parts:
                return "\n".join(text_parts).strip()

    return ""


class ExperimentRunner:
    """Encapsulates task and evaluators for Langfuse experiment runner."""

    def __init__(self):
        vector_store = VectorStore(
            qdrant_url=config.QDRANT_URL,
            qdrant_api_key=config.QDRANT_API_KEY,
            collection_name=config.QDRANT_COLLECTION_NAME,
        )
        retriever_tool = RetrieverTool(vector_store=vector_store)
        self._agent = Agent(vector_store=vector_store, retriever_tool=retriever_tool)

    def task(self, *, item, **kwargs) -> str:
        """Run repo's RAG agent on one dataset item."""
        question = _to_text(item.input)
        result = self._agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={
                "metadata": {
                    "langfuse_tags": ["experiment", "sdk", "telco"],
                    "experiment_source": "scripts/run_langfuse_experiment.py",
                }
            },
        )
        reply = _extract_reply(result)
        return reply or ""


def _set_if_supported(kwargs: dict[str, Any], fn, key: str, value: Any) -> None:
    """Set argument only if callable signature supports it."""
    if value is None:
        return
    try:
        if key in inspect.signature(fn).parameters:
            kwargs[key] = value
    except Exception:
        # If signature introspection fails, skip optional args.
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Langfuse experiment on existing dataset")
    parser.add_argument("--dataset", default="test", help="Existing Langfuse dataset name (default: test)")
    parser.add_argument("--name", default=None, help="Experiment run name (default: auto timestamp)")
    parser.add_argument(
        "--description",
        default="RAG agent evaluation run (scores configured in Langfuse UI evaluators)",
        help="Experiment description",
    )
    parser.add_argument("--max-concurrency", type=int, default=3, help="Max concurrent tasks if SDK supports it")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    langfuse = get_client()
    dataset = langfuse.get_dataset(args.dataset)

    run_name = args.name or f"telco-correctness-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    runner = ExperimentRunner()

    experiment_kwargs: dict[str, Any] = {
        "name": run_name,
        "description": args.description,
        "task": runner.task,
    }

    # Keep compatibility across minor SDK shape changes.
    _set_if_supported(experiment_kwargs, dataset.run_experiment, "max_concurrency", args.max_concurrency)

    result = dataset.run_experiment(**experiment_kwargs)

    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Experiment: {run_name}")
    print("=" * 80)
    print(result.format())

    # Best effort flush to deliver telemetry/scores before exit.
    try:
        langfuse.flush()
    except Exception:
        pass


if __name__ == "__main__":
    main()
