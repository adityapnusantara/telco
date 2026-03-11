from pydantic import BaseModel

class QNADocument(BaseModel):
    """Q&A pair document for knowledge base"""
    question: str
    answer: str
    source: str
    category: str
