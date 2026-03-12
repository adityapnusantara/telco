from unittest.mock import patch, MagicMock
from app.services.rag.ingestion import ingest_qna_documents

@patch('app.services.rag.ingestion.VectorStore')
def test_ingest_qna_documents(mock_vector_store_class):
    """Test ingesting Q&A documents into vector store"""
    mock_vs_instance = MagicMock()
    mock_vs_instance.store = MagicMock()
    mock_vector_store_class.return_value = mock_vs_instance

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

    mock_vs_instance.store.add_documents.assert_called_once()
    mock_vector_store_class.assert_called_once()
