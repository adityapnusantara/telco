from unittest.mock import patch, MagicMock
from app.services.rag.ingestion import ingest_qna_documents

@patch('app.services.rag.ingestion.get_vector_store')
def test_ingest_qna_documents(mock_get_vs):
    """Test ingesting Q&A documents into vector store"""
    mock_vs = MagicMock()
    mock_get_vs.return_value = mock_vs

    from app.services.rag.models import QNADocument

    documents = [
        QNADocument(
            question="Test question?",
            answer="Test answer.",
            source="test.md",
            category="test"
        )
    ]

    ingest_qna_documents(documents)

    mock_vs.add_documents.assert_called_once()
