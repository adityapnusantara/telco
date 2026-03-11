from app.services.rag.knowledge_base import load_qna_documents

def test_load_qna_documents():
    """Test loading Q&A documents from JSON files"""
    documents = load_qna_documents("data/kb")
    assert len(documents) > 0
    assert all(hasattr(doc, 'question') for doc in documents)
    assert all(hasattr(doc, 'answer') for doc in documents)
    assert all(hasattr(doc, 'source') for doc in documents)
    assert all(hasattr(doc, 'category') for doc in documents)
