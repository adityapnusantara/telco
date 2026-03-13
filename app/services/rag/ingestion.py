from pathlib import Path

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from app.core.config import config
from app.prompts.langfuse import get_extraction_prompt
from .knowledge_base import (
    clear_qna_json_directory,
    load_markdown_sources,
    load_qna_documents,
    save_qna_documents,
)
from .models import QNADocument, QNAExtractionResult
from .vector_store import VectorStore


def _build_extraction_agent():
    extraction_prompt = get_extraction_prompt()
    model_config = extraction_prompt["model_config"]

    llm = ChatOpenAI(
        model=model_config["model"],
        temperature=model_config["temperature"],
    )
    return create_agent(
        model=llm,
        tools=[],
        system_prompt=extraction_prompt["system_prompt"],
        response_format=QNAExtractionResult,
    ), extraction_prompt


def _extract_qna_for_source(
    agent,
    extraction_user_prompt,
    markdown_content: str,
    source: str,
    category: str,
) -> list[QNADocument]:
    compiled_prompt = extraction_user_prompt.compile(
        source=source,
        category=category,
        markdown_content=markdown_content,
    )
    result = agent.invoke({"messages": [{"role": "user", "content": compiled_prompt}]})
    structured = result.get("structured_response")
    if structured is None:
        raise ValueError(f"Extraction agent returned no structured response for source '{source}'")

    extracted = structured if isinstance(structured, QNAExtractionResult) else QNAExtractionResult.model_validate(structured)

    normalized_items: list[QNADocument] = []
    for item in extracted.items:
        normalized_items.append(
            item.model_copy(
                update={
                    "source": source,
                    "category": category,
                }
            )
        )
    return normalized_items


def extract_qna_from_markdown(kb_md_dir: str = "data/kb_md", kb_json_dir: str = "data/kb_json") -> int:
    """Extract Q&A pairs from markdown files and write JSON artifacts."""
    markdown_sources = load_markdown_sources(kb_md_dir)
    if not markdown_sources:
        raise ValueError(f"No markdown source files found in '{kb_md_dir}'")

    clear_qna_json_directory(kb_json_dir)
    extraction_agent, extraction_prompt = _build_extraction_agent()

    total_qna = 0
    for source_path, markdown_content in markdown_sources:
        source_name = source_path.name
        category = source_path.stem
        items = _extract_qna_for_source(
            extraction_agent,
            extraction_prompt["user_prompt"],
            markdown_content=markdown_content,
            source=source_name,
            category=category,
        )
        output_path = Path(kb_json_dir) / f"{source_path.stem}.json"
        save_qna_documents(output_path, items)
        total_qna += len(items)

    return total_qna


def ingest_qna_documents(documents: list[QNADocument]) -> None:
    """Ingest Q&A documents into the vector store."""
    vector_store = VectorStore(
        qdrant_url=config.QDRANT_URL,
        qdrant_api_key=config.QDRANT_API_KEY,
        collection_name=config.QDRANT_COLLECTION_NAME,
    )

    langchain_docs = []
    for qna in documents:
        content = f"Question: {qna.question}\nAnswer: {qna.answer}"
        doc = Document(
            page_content=content,
            metadata={
                "source": qna.source,
                "category": qna.category,
                "question": qna.question,
            },
        )
        langchain_docs.append(doc)

    vector_store.store.add_documents(langchain_docs)


def ingest_qna_json_directory(kb_json_dir: str = "data/kb_json") -> int:
    """Load generated Q&A JSON files and ingest into vector store."""
    documents = load_qna_documents(kb_json_dir)
    ingest_qna_documents(documents)
    return len(documents)


def run_full_ingestion(kb_md_dir: str = "data/kb_md", kb_json_dir: str = "data/kb_json") -> tuple[int, int]:
    """Run end-to-end ingestion: markdown extraction then vector ingestion."""
    extract_qna_from_markdown(kb_md_dir=kb_md_dir, kb_json_dir=kb_json_dir)
    generated_files = len(list(Path(kb_json_dir).glob("*.json")))
    ingested_documents = ingest_qna_json_directory(kb_json_dir=kb_json_dir)
    return generated_files, ingested_documents
