import json
from pathlib import Path
from .models import QNADocument


def load_markdown_sources(kb_md_dir: str = "data/kb_md") -> list[tuple[Path, str]]:
    """Load markdown KB files as (path, content) tuples."""
    kb_path = Path(kb_md_dir)
    sources: list[tuple[Path, str]] = []

    for md_file in sorted(kb_path.glob("*.md")):
        with open(md_file, "r", encoding="utf-8") as f:
            sources.append((md_file, f.read()))

    return sources


def save_qna_documents(json_path: Path, documents: list[QNADocument]) -> None:
    """Save Q&A documents to JSON file using overwrite behavior."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [doc.model_dump() for doc in documents]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def clear_qna_json_directory(kb_json_dir: str = "data/kb_json") -> None:
    """Clear existing generated JSON files before extraction."""
    kb_path = Path(kb_json_dir)
    kb_path.mkdir(parents=True, exist_ok=True)
    for json_file in kb_path.glob("*.json"):
        json_file.unlink()


def load_qna_documents(kb_json_dir: str = "data/kb_json") -> list[QNADocument]:
    """Load Q&A documents from generated JSON artifacts."""
    kb_path = Path(kb_json_dir)
    documents: list[QNADocument] = []

    for json_file in sorted(kb_path.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            qna_list = json.load(f)
            for qna_data in qna_list:
                documents.append(QNADocument(**qna_data))

    return documents
