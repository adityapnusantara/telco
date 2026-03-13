from pydantic import ValidationError
import pytest
from app.services.rag.models import QNADocument, QNAExtractionResult

def test_qna_document_valid():
    """Test creating a valid Q&A document"""
    qna = QNADocument(
        question="Test question?",
        answer="Test answer.",
        source="test.md",
        category="test"
    )
    assert qna.question == "Test question?"
    assert qna.answer == "Test answer."
    assert qna.source == "test.md"
    assert qna.category == "test"

def test_qna_document_missing_required_field():
    """Test that missing required fields raise validation error"""
    with pytest.raises(ValidationError):
        QNADocument(
            question="Test question?"
            # missing answer, source, category
        )


def test_qna_extraction_result_defaults_to_empty_items():
    """Test extraction result schema default values."""
    result = QNAExtractionResult()
    assert result.items == []
