from langchain_core.documents import Document
from .models import QNADocument
from .vector_store import get_vector_store

def ingest_qna_documents(documents: list[QNADocument]) -> None:
    """Ingest Q&A documents into the vector store"""
    vector_store = get_vector_store()

    langchain_docs = []
    for qna in documents:
        content = f"Question: {qna.question}\nAnswer: {qna.answer}"
        doc = Document(
            page_content=content,
            metadata={
                "source": qna.source,
                "category": qna.category,
                "question": qna.question,
            }
        )
        langchain_docs.append(doc)

    vector_store.add_documents(langchain_docs)

def ingest_from_directory(kb_dir: str = "data/kb") -> int:
    """Load and ingest Q&A documents from directory"""
    from .knowledge_base import load_qna_documents

    documents = load_qna_documents(kb_dir)
    ingest_qna_documents(documents)
    return len(documents)
