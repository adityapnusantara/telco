import json
from pathlib import Path
from typing import List
from .models import QNADocument

def load_qna_documents(kb_dir: str = "data/kb") -> List[QNADocument]:
    """
    Load Q&A documents from JSON files in the knowledge base directory.

    Args:
        kb_dir: Path to knowledge base directory

    Returns:
        List of QNADocument objects
    """
    kb_path = Path(kb_dir)
    documents = []

    for json_file in kb_path.glob("*.json"):
        with open(json_file, 'r') as f:
            qna_list = json.load(f)
            for qna_data in qna_list:
                documents.append(QNADocument(**qna_data))

    return documents
