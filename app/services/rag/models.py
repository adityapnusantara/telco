from pydantic import BaseModel, Field

class QNADocument(BaseModel):
    """Q&A pair document for knowledge base"""
    question: str
    answer: str
    source: str
    category: str


class QNAExtractionResult(BaseModel):
    """Structured output schema for markdown Q&A extraction."""
    items: list[QNADocument] = Field(default_factory=list)
