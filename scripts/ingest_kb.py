#!/usr/bin/env python
"""Script to ingest Q&A documents into the vector store."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.rag.ingestion import ingest_from_directory
from app.core.config import config

def main():
    """Main ingestion function"""
    print(f"Ingesting knowledge base from: data/kb")
    print(f"Qdrant collection: {config.QDRANT_COLLECTION_NAME}")

    count = ingest_from_directory("data/kb")

    print(f"✅ Successfully ingested {count} Q&A documents")

if __name__ == "__main__":
    main()
