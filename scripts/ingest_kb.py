#!/usr/bin/env python
"""Script to extract Q&A from markdown and ingest into the vector store."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.rag.ingestion import run_full_ingestion
from app.core.config import config

def main():
    """Main ingestion function"""
    print("Extracting markdown knowledge base from: data/kb_md")
    print("Generated Q&A JSON output directory: data/kb_json")
    print(f"Qdrant collection: {config.QDRANT_COLLECTION_NAME}")

    generated_files, ingested_count = run_full_ingestion("data/kb_md", "data/kb_json")

    print(f"✅ Generated {generated_files} JSON files from markdown sources")
    print(f"✅ Successfully ingested {ingested_count} Q&A documents")

if __name__ == "__main__":
    main()
