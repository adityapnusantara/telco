from app.services.rag.knowledge_base import load_markdown_sources, load_qna_documents

def test_load_qna_documents():
    """Test loading Q&A documents from JSON files"""
    documents = load_qna_documents("data/kb_json")
    assert len(documents) > 0
    assert all(hasattr(doc, 'question') for doc in documents)
    assert all(hasattr(doc, 'answer') for doc in documents)
    assert all(hasattr(doc, 'source') for doc in documents)
    assert all(hasattr(doc, 'category') for doc in documents)


def test_load_markdown_sources():
    """Test loading markdown source documents"""
    sources = load_markdown_sources("data/kb_md")
    assert len(sources) == 3
    assert all(path.suffix == ".md" for path, _ in sources)
    assert all(content.strip() for _, content in sources)
