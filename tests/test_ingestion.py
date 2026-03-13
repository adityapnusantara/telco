import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.services.rag.ingestion import (
    extract_qna_from_markdown,
    ingest_qna_documents,
    run_full_ingestion,
)
from app.services.rag.models import QNADocument, QNAExtractionResult


@patch("app.services.rag.ingestion.VectorStore")
def test_ingest_qna_documents(mock_vector_store_class):
    """Test ingesting Q&A documents into vector store."""
    mock_vs_instance = MagicMock()
    mock_vs_instance.store = MagicMock()
    mock_vector_store_class.return_value = mock_vs_instance

    documents = [
        QNADocument(
            question="Test question?",
            answer="Test answer.",
            source="test.md",
            category="test",
        )
    ]

    ingest_qna_documents(documents)

    mock_vs_instance.store.add_documents.assert_called_once()
    mock_vector_store_class.assert_called_once()


def test_extract_qna_from_markdown_overwrites_json(tmp_path):
    """Extraction should overwrite old JSON artifacts and write new content."""
    kb_md_dir = tmp_path / "kb_md"
    kb_json_dir = tmp_path / "kb_json"
    kb_md_dir.mkdir()
    kb_json_dir.mkdir()

    (kb_md_dir / "billing_policy.md").write_text("# Billing\n- Bills are generated monthly.", encoding="utf-8")
    stale_json = kb_json_dir / "billing_policy.json"
    stale_json.write_text(
        json.dumps(
            [
                {
                    "question": "old",
                    "answer": "old",
                    "source": "old.md",
                    "category": "old",
                }
            ]
        ),
        encoding="utf-8",
    )

    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {
        "structured_response": QNAExtractionResult(
            items=[
                QNADocument(
                    question="When are bills generated?",
                    answer="Bills are generated monthly.",
                    source="ignored.md",
                    category="ignored",
                )
            ]
        )
    }
    mock_user_prompt = MagicMock()
    mock_user_prompt.compile.return_value = "Compiled extraction prompt"

    with patch("app.services.rag.ingestion.create_agent", return_value=mock_agent), patch(
        "app.services.rag.ingestion.ChatOpenAI"
    ), patch(
        "app.services.rag.ingestion.get_extraction_prompt",
        return_value={
            "system_prompt": "Extraction system prompt",
            "user_prompt": mock_user_prompt,
            "model_config": {"model": "gpt-4o-mini", "temperature": 0},
        },
    ):
        total = extract_qna_from_markdown(str(kb_md_dir), str(kb_json_dir))

    assert total == 1
    mock_user_prompt.compile.assert_called_once_with(
        source="billing_policy.md",
        category="billing_policy",
        markdown_content="# Billing\n- Bills are generated monthly.",
    )
    payload = json.loads(stale_json.read_text(encoding="utf-8"))
    assert payload[0]["question"] == "When are bills generated?"
    assert payload[0]["source"] == "billing_policy.md"
    assert payload[0]["category"] == "billing_policy"


def test_extract_qna_from_markdown_raises_when_no_sources(tmp_path):
    """Extraction should fail fast when no markdown files exist."""
    kb_md_dir = tmp_path / "kb_md"
    kb_json_dir = tmp_path / "kb_json"
    kb_md_dir.mkdir()
    kb_json_dir.mkdir()

    with pytest.raises(ValueError, match="No markdown source files found"):
        extract_qna_from_markdown(str(kb_md_dir), str(kb_json_dir))


@patch("app.services.rag.ingestion.ingest_qna_json_directory", return_value=4)
def test_run_full_ingestion_orchestrates_extract_then_ingest(mock_ingest_json, tmp_path):
    """Full ingestion should extract markdown first then ingest JSON."""
    kb_md_dir = tmp_path / "kb_md"
    kb_json_dir = tmp_path / "kb_json"
    kb_md_dir.mkdir()
    kb_json_dir.mkdir()

    (kb_md_dir / "service_plans.md").write_text("# Plans\n- Basic plan", encoding="utf-8")

    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {
        "structured_response": QNAExtractionResult(
            items=[
                QNADocument(
                    question="What is the basic plan?",
                    answer="Basic plan is available.",
                    source="x.md",
                    category="x",
                )
            ]
        )
    }
    mock_user_prompt = MagicMock()
    mock_user_prompt.compile.return_value = "Compiled extraction prompt"

    with patch("app.services.rag.ingestion.create_agent", return_value=mock_agent), patch(
        "app.services.rag.ingestion.ChatOpenAI"
    ), patch(
        "app.services.rag.ingestion.get_extraction_prompt",
        return_value={
            "system_prompt": "Extraction system prompt",
            "user_prompt": mock_user_prompt,
            "model_config": {"model": "gpt-4o-mini", "temperature": 0},
        },
    ):
        generated_files, ingested_documents = run_full_ingestion(str(kb_md_dir), str(kb_json_dir))

    assert generated_files == 1
    assert ingested_documents == 4
    mock_user_prompt.compile.assert_called_once_with(
        source="service_plans.md",
        category="service_plans",
        markdown_content="# Plans\n- Basic plan",
    )
    mock_ingest_json.assert_called_once_with(kb_json_dir=str(kb_json_dir))
    assert Path(kb_json_dir / "service_plans.json").exists()
