# Telco Customer Service AI Agent - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a working Customer Service AI Agent with RAG pipeline for a Telco company, using LangChain, OpenAI GPT-4o, Qdrant Cloud, and Langfuse for prompt management and tracing.

**Architecture:** FastAPI service with LangChain `create_agent` using a retriever tool. System prompts managed in Langfuse, Q&A pairs stored in Qdrant Cloud, all traces logged via Langfuse CallbackHandler.

**Tech Stack:** FastAPI, LangChain 1.2.11, OpenAI GPT-4o, Qdrant Cloud, Langfuse 4.0.0, Python 3.11+

---

## Task 1: Project Setup - pyproject.toml

**Files:**
- Modify: `pyproject.toml`

**Step 1: Update pyproject.toml with all dependencies**

```toml
[tool.poetry]
name = "telco"
version = "0.1.0"
description = "Telco Customer Service AI Agent"
authors = ["AI Engineer"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.135.1"
uvicorn = {extras = ["standard"], version = "^0.41.0"}
pydantic = "^2.12.5"
python-dotenv = "^1.0.0"

# LangChain
langchain = "^1.2.11"
langchain-openai = "^1.1.11"
langchain-qdrant = "^1.1.0"
langchain-text-splitters = "^1.1.1"

# Langfuse
langfuse = "^4.0.0"

# Qdrant Client
qdrant-client = "^1.17.0"

[tool.poetry.group.dev.dependencies]
pytest = "^8.3.0"
httpx = "^0.27.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

**Step 2: Install dependencies**

Run: `poetry install`
Expected: Dependencies installed successfully

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add project dependencies"
```

---

## Task 2: Environment Configuration

**Files:**
- Create: `.env.example`

**Step 1: Create .env.example**

```bash
OPENAI_API_KEY=sk-proj-...

# Langfuse Configuration
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com

# Qdrant Cloud Configuration
QDRANT_URL=https://cfcd371f-c4ef-4a64-99c3-243b273a5f07.europe-west3-0.gcp.cloud.qdrant.io
QDRANT_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.oSwLG6-eGwC2afvtfcX-vyhWt85qMuSyfL3ee_imJfc

# Application Configuration
APP_ENV=development
LOG_LEVEL=info
```

**Step 2: Commit**

```bash
git add .env.example
git commit -m "chore: add environment configuration template"
```

---

## Task 3: Core Configuration Module

**Files:**
- Create: `app/core/config.py`
- Create: `app/core/__init__.py`

**Step 1: Write the test for configuration loading**

Create `tests/test_config.py`:

```python
import os
from dotenv import load_dotenv

def test_config_can_be_loaded():
    """Test that environment variables can be loaded"""
    load_dotenv()
    assert os.getenv("OPENAI_API_KEY") is not None or os.getenv("OPENAI_API_KEY", "") == "sk-proj-..."
    assert os.getenv("LANGFUSE_PUBLIC_KEY") is not None
    assert os.getenv("LANGFUSE_SECRET_KEY") is not None
    assert os.getenv("QDRANT_URL") is not None
    assert os.getenv("QDRANT_API_KEY") is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_config_can_be_loaded -v`
Expected: FAIL or PASS (if .env exists)

**Step 3: Create config module**

Create `app/core/__init__.py` (empty file)

Create `app/core/config.py`:

```python
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Langfuse
    LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    LANGFUSE_BASE_URL: str = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

    # Qdrant
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION_NAME: str = "telco_knowledge_base"

    # App
    APP_ENV: str = os.getenv("APP_ENV", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")

config = Config()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py::test_config_can_be_loaded -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/core/ tests/test_config.py
git commit -m "feat: add core configuration module"
```

---

## Task 4: Knowledge Base Data Files

**Files:**
- Create: `data/kb/billing_qna.json`
- Create: `data/kb/plans_qna.json`
- Create: `data/kb/troubleshooting_qna.json`

**Step 1: Create billing Q&A data**

Create `data/kb/billing_qna.json`:

```json
[
  {
    "question": "When are bills generated?",
    "answer": "Bills are generated on the 1st of every month.",
    "source": "billing_policy.md",
    "category": "billing"
  },
  {
    "question": "What is the late payment fee?",
    "answer": "A late payment fee of IDR 50,000 applies after 14 days overdue.",
    "source": "billing_policy.md",
    "category": "billing"
  },
  {
    "question": "How can I request a billing dispute?",
    "answer": "Customers can request a billing dispute within 30 days of the invoice date.",
    "source": "billing_policy.md",
    "category": "billing"
  },
  {
    "question": "How do I enroll in auto-pay?",
    "answer": "Auto-pay enrollment is available via the MyTelco app.",
    "source": "billing_policy.md",
    "category": "billing"
  }
]
```

**Step 2: Create service plans Q&A data**

Create `data/kb/plans_qna.json`:

```json
[
  {
    "question": "What is the Basic Plan?",
    "answer": "Basic Plan: IDR 99,000/month with 10GB data and unlimited calls.",
    "source": "service_plans.md",
    "category": "plans"
  },
  {
    "question": "What is the Pro Plan?",
    "answer": "Pro Plan: IDR 199,000/month with 50GB data, unlimited calls, and 5GB hotspot.",
    "source": "service_plans.md",
    "category": "plans"
  },
  {
    "question": "What is the Unlimited Plan?",
    "answer": "Unlimited Plan: IDR 299,000/month with unlimited data, calls, and 20GB hotspot.",
    "source": "service_plans.md",
    "category": "plans"
  },
  {
    "question": "Do all plans include streaming access?",
    "answer": "Yes, all plans include free access to streaming partners on weekends.",
    "source": "service_plans.md",
    "category": "plans"
  }
]
```

**Step 3: Create troubleshooting Q&A data**

Create `data/kb/troubleshooting_qna.json`:

```json
[
  {
    "question": "What should I do for slow internet?",
    "answer": "For slow internet: restart the device, check signal strength, toggle airplane mode.",
    "source": "troubleshooting_guide.md",
    "category": "troubleshooting"
  },
  {
    "question": "How do I fix call quality issues?",
    "answer": "For call quality issues: check for network congestion via the MyTelco app.",
    "source": "troubleshooting_guide.md",
    "category": "troubleshooting"
  },
  {
    "question": "How do I report billing errors?",
    "answer": "For billing errors: submit a ticket via the app or call 123 (free from any network).",
    "source": "troubleshooting_guide.md",
    "category": "troubleshooting"
  },
  {
    "question": "How do I get a SIM card replacement?",
    "answer": "SIM card replacement is available at any authorized store with valid ID.",
    "source": "troubleshooting_guide.md",
    "category": "troubleshooting"
  }
]
```

**Step 4: Commit**

```bash
git add data/kb/
git commit -m "feat: add knowledge base Q&A data files"
```

---

## Task 5: Q&A Document Model

**Files:**
- Create: `app/services/rag/models.py`
- Create: `app/services/rag/__init__.py`

**Step 1: Write the test for Q&A model**

Create `tests/test_qna_model.py`:

```python
from pydantic import ValidationError
import pytest
from app.services.rag.models import QNADocument

def test_qna_document_valid():
    """Test creating a valid Q&A document"""
    qna = QNADocument(
        question="Test question?",
        answer="Test answer.",
        source="test.md",
        category="test"
    )
    assert qna.question == "Test question?"
    assert qna.answer == "Test answer."
    assert qna.source == "test.md"
    assert qna.category == "test"

def test_qna_document_missing_required_field():
    """Test that missing required fields raise validation error"""
    with pytest.raises(ValidationError):
        QNADocument(
            question="Test question?"
            # missing answer, source, category
        )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_qna_model.py -v`
Expected: FAIL - "ModuleNotFoundError: No module named 'app.services.rag.models'"

**Step 3: Create Q&A model**

Create `app/services/rag/__init__.py` (empty file)

Create `app/services/rag/models.py`:

```python
from pydantic import BaseModel

class QNADocument(BaseModel):
    """Q&A pair document for knowledge base"""
    question: str
    answer: str
    source: str
    category: str
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_qna_model.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/services/rag/ tests/test_qna_model.py
git commit -m "feat: add Q&A document model"
```

---

## Task 6: Knowledge Base Manager

**Files:**
- Create: `app/services/rag/knowledge_base.py`

**Step 1: Write the test for loading Q&A files**

Create `tests/test_knowledge_base.py`:

```python
from app.services.rag.knowledge_base import load_qna_documents

def test_load_qna_documents():
    """Test loading Q&A documents from JSON files"""
    documents = load_qna_documents("data/kb")
    assert len(documents) > 0
    assert all(hasattr(doc, 'question') for doc in documents)
    assert all(hasattr(doc, 'answer') for doc in documents)
    assert all(hasattr(doc, 'source') for doc in documents)
    assert all(hasattr(doc, 'category') for doc in documents)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_knowledge_base.py::test_load_qna_documents -v`
Expected: FAIL - "ModuleNotFoundError: No module named 'app.services.rag.knowledge_base'"

**Step 3: Implement knowledge base loader**

Create `app/services/rag/knowledge_base.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_knowledge_base.py::test_load_qna_documents -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/services/rag/knowledge_base.py tests/test_knowledge_base.py
git commit -m "feat: add knowledge base loader"
```

---

## Task 7: Langfuse Integration

**Files:**
- Create: `app/prompts/langfuse.py`
- Create: `app/prompts/__init__.py`

**Step 1: Write the test for Langfuse client initialization**

Create `tests/test_langfuse.py`:

```python
from unittest.mock import patch, MagicMock
from app.prompts.langfuse import get_langfuse_client, get_system_prompt

def test_get_langfuse_client():
    """Test Langfuse client initialization"""
    with patch('app.prompts.langfuse.get_client') as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        client = get_langfuse_client()

        assert client is not None
        mock_get_client.assert_called_once()

@patch('app.prompts.langfuse.get_client')
def test_get_system_prompt(mock_get_client):
    """Test fetching system prompt from Langfuse"""
    mock_client = MagicMock()
    mock_prompt = MagicMock()
    mock_prompt.compile.return_value = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    mock_client.get_prompt.return_value = mock_prompt
    mock_get_client.return_value = mock_client

    prompt = get_system_prompt("telco-customer-service-agent")

    assert prompt == [{"role": "system", "content": "You are a helpful assistant."}]
    mock_client.get_prompt.assert_called_once_with(
        "telco-customer-service-agent",
        type="chat"
    )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_langfuse.py -v`
Expected: FAIL - "ModuleNotFoundError: No module named 'app.prompts.langfuse'"

**Step 3: Implement Langfuse integration**

Create `app/prompts/__init__.py` (empty file)

Create `app/prompts/langfuse.py`:

```python
from langfuse import get_client
from app.core.config import config

# Initialize Langfuse client once at module level
_langfuse_client = None

def get_langfuse_client():
    """
    Get or create the Langfuse client singleton.

    Returns:
        Langfuse client instance
    """
    global _langfuse_client
    if _langfuse_client is None:
        _langfuse_client = get_client()
    return _langfuse_client

def get_system_prompt(prompt_name: str = "telco-customer-service-agent"):
    """
    Fetch the system prompt from Langfuse Prompt Management.

    Args:
        prompt_name: Name of the prompt in Langfuse

    Returns:
        List of message dicts for LangChain
    """
    client = get_langfuse_client()
    prompt = client.get_prompt(prompt_name, type="chat")

    # Compile with default values
    compiled = prompt.compile(
        company_name="MyTelco",
        escalation_contact="call 123 or use the MyTelco app"
    )

    return compiled
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_langfuse.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/prompts/ tests/test_langfuse.py
git commit -m "feat: add Langfuse integration"
```

---

## Task 8: Vector Store Setup

**Files:**
- Create: `app/services/rag/vector_store.py`

**Step 1: Write the test for vector store initialization**

Create `tests/test_vector_store.py`:

```python
from unittest.mock import patch, MagicMock
from app.services.rag.vector_store import get_vector_store

@patch('app.services.rag.vector_store.QdrantVectorStore')
@patch('app.services.rag.vector_store.OpenAIEmbeddings')
@patch('app.services.rag.vector_store.QdrantClient')
def test_get_vector_store(mock_qdrant_client, mock_embeddings, mock_vector_store):
    """Test vector store initialization"""
    mock_client_instance = MagicMock()
    mock_qdrant_client.return_value = mock_client_instance

    mock_embeddings_instance = MagicMock()
    mock_embeddings.return_value = mock_embeddings_instance

    mock_vs_instance = MagicMock()
    mock_vector_store.from_existing_collection.return_value = mock_vs_instance

    vs = get_vector_store()

    assert vs is not None
    mock_vector_store.from_existing_collection.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vector_store.py::test_get_vector_store -v`
Expected: FAIL - "ModuleNotFoundError: No module named 'app.services.rag.vector_store'"

**Step 3: Implement vector store setup**

Create `app/services/rag/vector_store.py`:

```python
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from app.core.config import config

# Singleton instances
_qdrant_client = None
_vector_store = None

def get_qdrant_client():
    """Get or create Qdrant client singleton"""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
        )
    return _qdrant_client

def get_vector_store():
    """
    Get or create Qdrant vector store singleton.

    Returns:
        QdrantVectorStore instance
    """
    global _vector_store
    if _vector_store is None:
        client = get_qdrant_client()
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=512,
        )

        _vector_store = QdrantVectorStore(
            client=client,
            collection_name=config.QDRANT_COLLECTION_NAME,
            embedding=embeddings,
        )
    return _vector_store
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vector_store.py::test_get_vector_store -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/services/rag/vector_store.py tests/test_vector_store.py
git commit -m "feat: add vector store setup"
```

---

## Task 9: Document Ingestion

**Files:**
- Create: `app/services/rag/ingestion.py`

**Step 1: Write the test for document ingestion**

Create `tests/test_ingestion.py`:

```python
from unittest.mock import patch, MagicMock
from app.services.rag.ingestion import ingest_qna_documents

@patch('app.services.rag.ingestion.get_vector_store')
def test_ingest_qna_documents(mock_get_vs):
    """Test ingesting Q&A documents into vector store"""
    mock_vs = MagicMock()
    mock_get_vs.return_value = mock_vs

    from app.services.rag.models import QNADocument

    documents = [
        QNADocument(
            question="Test question?",
            answer="Test answer.",
            source="test.md",
            category="test"
        )
    ]

    ingest_qna_documents(documents)

    mock_vs.add_documents.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingestion.py::test_ingest_qna_documents -v`
Expected: FAIL - "ModuleNotFoundError: No module named 'app.services.rag.ingestion'"

**Step 3: Implement document ingestion**

Create `app/services/rag/ingestion.py`:

```python
from langchain_core.documents import Document
from .models import QNADocument
from .vector_store import get_vector_store

def ingest_qna_documents(documents: list[QNADocument]) -> None:
    """
    Ingest Q&A documents into the vector store.

    Args:
        documents: List of QNADocument objects
    """
    vector_store = get_vector_store()

    # Convert Q&A documents to LangChain Documents
    langchain_docs = []
    for qna in documents:
        # Combine question and answer for better retrieval
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

    # Add to vector store
    vector_store.add_documents(langchain_docs)

def ingest_from_directory(kb_dir: str = "data/kb") -> int:
    """
    Load and ingest Q&A documents from directory.

    Args:
        kb_dir: Path to knowledge base directory

    Returns:
        Number of documents ingested
    """
    from .knowledge_base import load_qna_documents

    documents = load_qna_documents(kb_dir)
    ingest_qna_documents(documents)
    return len(documents)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingestion.py::test_ingest_qna_documents -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/services/rag/ingestion.py tests/test_ingestion.py
git commit -m "feat: add document ingestion"
```

---

## Task 10: Retriever Tool

**Files:**
- Create: `app/services/rag/retriever.py`

**Step 1: Write the test for retriever**

Create `tests/test_retriever.py`:

```python
from unittest.mock import patch, MagicMock
from app.services.rag.retriever import get_retriever_tool

@patch('app.services.rag.retriever.get_vector_store')
def test_get_retriever_tool(mock_get_vs):
    """Test creating retriever tool"""
    mock_vs = MagicMock()
    mock_vs.as_retriever.return_value = MagicMock()
    mock_get_vs.return_value = mock_vs

    tool = get_retriever_tool()

    assert tool is not None
    assert tool.name == "search_knowledge_base"
    assert "knowledge base" in tool.description.lower()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_retriever.py::test_get_retriever_tool -v`
Expected: FAIL - "ModuleNotFoundError: No module named 'app.services.rag.retriever'"

**Step 3: Implement retriever tool**

Create `app/services/rag/retriever.py`:

```python
from langchain_core.tools import tool
from .vector_store import get_vector_store

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Telco knowledge base for relevant information.

    Args:
        query: The user's question or search query

    Returns:
        Relevant information from the knowledge base
    """
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information found in the knowledge base."

    # Format results
    results = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        category = doc.metadata.get("category", "general")
        results.append(f"[{category} - {source}]: {doc.page_content}")

    return "\n\n".join(results)

def get_retriever_tool():
    """Get the retriever tool for use with LangChain agents"""
    return search_knowledge_base
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_retriever.py::test_get_retriever_tool -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/services/rag/retriever.py tests/test_retriever.py
git commit -m "feat: add retriever tool"
```

---

## Task 11: LLM Agent Service

**Files:**
- Create: `app/services/llm/agent.py`
- Create: `app/services/llm/__init__.py`

**Step 1: Write the test for agent creation**

Create `tests/test_agent.py`:

```python
from unittest.mock import patch, MagicMock
from app.services.llm.agent import get_agent

@patch('app.services.llm.agent.get_system_prompt')
@patch('app.services.llm.agent.ChatOpenAI')
@patch('app.services.llm.agent.create_agent')
def test_get_agent(mock_create_agent, mock_chat_openai, mock_get_prompt):
    """Test agent creation"""
    mock_llm = MagicMock()
    mock_chat_openai.return_value = mock_llm

    mock_agent = MagicMock()
    mock_create_agent.return_value = mock_agent

    mock_get_prompt.return_value = [
        {"role": "system", "content": "Test prompt"}
    ]

    agent = get_agent()

    assert agent is not None
    mock_create_agent.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent.py::test_get_agent -v`
Expected: FAIL - "ModuleNotFoundError: No module named 'app.services.llm.agent'"

**Step 3: Implement agent service**

Create `app/services/llm/__init__.py` (empty file)

Create `app/services/llm/agent.py`:

```python
from langchain import create_agent
from langchain_openai import ChatOpenAI
from app.prompts.langfuse import get_system_prompt
from app.services.rag.retriever import get_retriever_tool

# Singleton agent
_agent = None

def get_agent():
    """
    Get or create the LangChain agent singleton.

    Returns:
        LangChain agent instance
    """
    global _agent
    if _agent is None:
        # Initialize LLM
        llm = ChatOpenAI(model="gpt-4o", temperature=0)

        # Get system prompt from Langfuse
        system_prompt = get_system_prompt()

        # Build prompt string from messages
        prompt_text = "\n".join([msg["content"] for msg in system_prompt])

        # Get retriever tool
        retriever_tool = get_retriever_tool()

        # Create agent
        _agent = create_agent(
            model=llm,
            tools=[retriever_tool],
            system_prompt=prompt_text,
        )

    return _agent
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent.py::test_get_agent -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/services/llm/ tests/test_agent.py
git commit -m "feat: add LLM agent service"
```

---

## Task 12: Langfuse Callback Handler

**Files:**
- Create: `app/services/llm/callbacks.py`

**Step 1: Write the test for callback handler**

Create `tests/test_callbacks.py`:

```python
from unittest.mock import patch
from app.services.llm.callbacks import get_langfuse_handler

@patch('app.services.llm.callbacks.CallbackHandler')
def test_get_langfuse_handler(mock_callback_handler):
    """Test Langfuse callback handler creation"""
    mock_handler = MagicMock()
    mock_callback_handler.return_value = mock_handler

    handler = get_langfuse_handler()

    assert handler is not None
    mock_callback_handler.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_callbacks.py::test_get_langfuse_handler -v`
Expected: FAIL - "ModuleNotFoundError: No module named 'app.services.llm.callbacks'"

**Step 3: Implement callback handler**

Create `app/services/llm/callbacks.py`:

```python
from langfuse.langchain import CallbackHandler

# Singleton handler
_handler = None

def get_langfuse_handler():
    """
    Get or create the Langfuse callback handler singleton.

    Returns:
        CallbackHandler instance
    """
    global _handler
    if _handler is None:
        _handler = CallbackHandler()
    return _handler
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_callbacks.py::test_get_langfuse_handler -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/services/llm/callbacks.py tests/test_callbacks.py
git commit -m "feat: add Langfuse callback handler"
```

---

## Task 13: Chat Service

**Files:**
- Create: `app/services/llm/chat.py`

**Step 1: Write the test for chat service**

Create `tests/test_chat_service.py`:

```python
from unittest.mock import patch, MagicMock
from app.services.llm.chat import chat, ChatResponse

@patch('app.services.llm.chat.get_langfuse_handler')
@patch('app.services.llm.chat.get_agent')
def test_chat_success(mock_get_agent, mock_get_handler):
    """Test successful chat response"""
    mock_agent = MagicMock()
    mock_response = MagicMock()
    mock_response.messages = [
        {"role": "assistant", "content": "Test response"}
    ]
    mock_agent.invoke.return_value = mock_response
    mock_get_agent.return_value = mock_agent

    mock_handler = MagicMock()
    mock_get_handler.return_value = mock_handler

    response = chat("Hello", [], None)

    assert isinstance(response, ChatResponse)
    assert response.reply == "Test response"
    assert response.escalate == False

@patch('app.services.llm.chat.get_agent')
def test_chat_escalation_on_empty_response(mock_get_agent):
    """Test escalation when agent cannot help"""
    mock_agent = MagicMock()
    mock_response = MagicMock()
    mock_response.messages = [
        {"role": "assistant", "content": "I cannot help with that."}
    ]
    mock_agent.invoke.return_value = mock_response
    mock_get_agent.return_value = mock_agent

    response = chat("What about Mars weather?", [], None)

    assert isinstance(response, ChatResponse)
    # Should check if escalation is triggered
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_chat_service.py -v`
Expected: FAIL - "ModuleNotFoundError: No module named 'app.services.llm.chat'"

**Step 3: Implement chat service**

Create `app/services/llm/chat.py`:

```python
from pydantic import BaseModel
from typing import Optional
from .agent import get_agent
from .callbacks import get_langfuse_handler

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    reply: str
    escalate: bool
    sources: Optional[list[str]] = None
    confidence_score: Optional[float] = None

def chat(
    message: str,
    conversation_history: list[dict],
    conversation_id: Optional[str] = None
) -> ChatResponse:
    """
    Process a chat message using the RAG agent.

    Args:
        message: User's message
        conversation_history: Previous messages in the conversation
        conversation_id: Optional conversation session ID

    Returns:
        ChatResponse with reply and escalation flag
    """
    agent = get_agent()
    handler = get_langfuse_handler()

    # Build messages list
    messages = conversation_history.copy()
    messages.append({"role": "user", "content": message})

    # Invoke agent with callbacks
    config = {
        "callbacks": [handler],
        "metadata": {
            "langfuse_session_id": conversation_id or "default",
            "langfuse_tags": ["chat", "telco-agent"]
        }
    }

    result = agent.invoke({"messages": messages}, config=config)

    # Extract response
    reply = result.messages[-1]["content"]

    # Determine escalation (simple heuristic)
    escalate = _should_escalate(reply)

    return ChatResponse(
        reply=reply,
        escalate=escalate,
    )

def _should_escalate(reply: str) -> bool:
    """
    Determine if the conversation should be escalated to a human.

    Args:
        reply: The agent's response

    Returns:
        True if escalation is needed
    """
    escalation_indicators = [
        "cannot help",
        "don't know",
        "unable to assist",
        "speak to a human",
        "transfer to agent",
        "escalate"
    ]

    reply_lower = reply.lower()
    return any(indicator in reply_lower for indicator in escalation_indicators)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_chat_service.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/services/llm/chat.py tests/test_chat_service.py
git commit -m "feat: add chat service"
```

---

## Task 14: API Models

**Files:**
- Create: `app/api/models.py`
- Create: `app/api/__init__.py`

**Step 1: Write the test for API models**

Create `tests/test_api_models.py`:

```python
from pydantic import ValidationError
import pytest
from app.api.models import ChatRequest, ChatResponse

def test_chat_request_valid():
    """Test creating a valid chat request"""
    req = ChatRequest(
        message="What are your plans?",
        conversation_history=[],
        conversation_id=None
    )
    assert req.message == "What are your plans?"
    assert req.conversation_history == []

def test_chat_request_minimal():
    """Test chat request with only required field"""
    req = ChatRequest(message="Test")
    assert req.message == "Test"

def test_chat_response_valid():
    """Test creating a valid chat response"""
    resp = ChatResponse(
        reply="Here are our plans...",
        escalate=False
    )
    assert resp.reply == "Here are our plans..."
    assert resp.escalate == False
    assert resp.sources is None

def test_chat_request_empty_message():
    """Test that empty message raises validation error"""
    with pytest.raises(ValidationError):
        ChatRequest(message="")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_api_models.py -v`
Expected: FAIL - "ModuleNotFoundError: No module named 'app.api.models'"

**Step 3: Implement API models**

Create `app/api/__init__.py` (empty file)

Create `app/api/models.py`:

```python
from pydantic import BaseModel, Field
from typing import Optional, List

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., min_length=1, description="User's message")
    conversation_id: Optional[str] = Field(None, description="Conversation session ID")
    conversation_history: Optional[List[dict]] = Field(
        default_factory=list,
        description="Previous messages in the conversation"
    )

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    reply: str = Field(..., description="Agent's response")
    escalate: bool = Field(..., description="Whether to escalate to human")
    sources: Optional[List[str]] = Field(None, description="Retrieved document names")
    confidence_score: Optional[float] = Field(None, description="Confidence score (0-1)")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_api_models.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/api/ tests/test_api_models.py
git commit -m "feat: add API models"
```

---

## Task 15: Chat Route

**Files:**
- Create: `app/api/routes/chat.py`
- Create: `app/api/routes/__init__.py`

**Step 1: Write the test for chat endpoint**

Create `tests/test_chat_endpoint.py`:

```python
from fastapi.testclient import TestClient
from unittest.mock import patch

def test_chat_endpoint_success(client):
    """Test successful chat request"""
    with patch('app.api.routes.chat.chat') as mock_chat:
        from app.services.llm.chat import ChatResponse
        mock_chat.return_value = ChatResponse(
            reply="Test response",
            escalate=False
        )

        response = client.post("/chat", json={
            "message": "What are your plans?",
            "conversation_history": []
        })

        assert response.status_code == 200
        data = response.json()
        assert data["reply"] == "Test response"
        assert data["escalate"] == False

def test_chat_endpoint_empty_message(client):
    """Test that empty message returns 422"""
    response = client.post("/chat", json={
        "message": "",
        "conversation_history": []
    })
    assert response.status_code == 422

def test_chat_endpoint_with_history(client):
    """Test chat endpoint with conversation history"""
    with patch('app.api.routes.chat.chat') as mock_chat:
        from app.services.llm.chat import ChatResponse
        mock_chat.return_value = ChatResponse(
            reply="Test response",
            escalate=False
        )

        response = client.post("/chat", json={
            "message": "And the Pro plan?",
            "conversation_history": [
                {"role": "user", "content": "What are your plans?"},
                {"role": "assistant", "content": "We have Basic, Pro, and Unlimited."}
            ]
        })

        assert response.status_code == 200
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_chat_endpoint.py -v`
Expected: FAIL - "ModuleNotFoundError: No module named 'app.api.routes.chat'"

**Step 3: Implement chat route**

Create `app/api/routes/__init__.py` (empty file)

Create `app/api/routes/chat.py`:

```python
from fastapi import APIRouter, HTTPException
from app.api.models import ChatRequest, ChatResponse
from app.services.llm.chat import chat as chat_service

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def create_chat(request: ChatRequest) -> ChatResponse:
    """
    Chat endpoint for the Telco customer service agent.

    Args:
        request: Chat request with message and optional history

    Returns:
        ChatResponse with reply and escalation flag
    """
    try:
        response = chat_service(
            message=request.message,
            conversation_history=request.conversation_history or [],
            conversation_id=request.conversation_id
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_chat_endpoint.py -v`
Expected: FAIL - Need to create conftest.py for client fixture

**Step 5: Create test fixtures**

Create `tests/conftest.py`:

```python
import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    """Test client for FastAPI app"""
    from app.main import app
    return TestClient(app)
```

**Step 6: Run test to verify it passes**

Run: `pytest tests/test_chat_endpoint.py -v`
Expected: FAIL - "ModuleNotFoundError: No module named 'app.main'"

**Step 7: Commit (will pass after main.py is created)**

```bash
git add app/api/routes/chat.py app/api/routes/__init__.py tests/test_chat_endpoint.py tests/conftest.py
git commit -m "feat: add chat route"
```

---

## Task 16: Main Application

**Files:**
- Create: `app/main.py`
- Create: `app/__init__.py`

**Step 1: Write the test for app creation**

Create `tests/test_main.py`:

```python
from fastapi.testclient import TestClient

def test_app_exists():
    """Test that the FastAPI app can be imported"""
    from app.main import app
    assert app is not None

def test_root_endpoint(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Telco Customer Service AI Agent"

def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_main.py -v`
Expected: FAIL - "ModuleNotFoundError: No module named 'app.main'"

**Step 3: Implement main application**

Create `app/__init__.py` (empty file)

Create `app/main.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes.chat import router as chat_router

app = FastAPI(
    title="Telco Customer Service AI Agent",
    description="AI-powered customer service agent for Telco",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router, tags=["chat"])

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Telco Customer Service AI Agent",
        "version": "0.1.0",
        "endpoints": {
            "chat": "/chat",
            "health": "/health"
        }
    }

@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "healthy"}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_main.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/ tests/test_main.py
git commit -m "feat: add main FastAPI application"
```

---

## Task 17: Application Entry Point

**Files:**
- Create: `run.py`

**Step 1: Create run script**

Create `run.py`:

```python
import uvicorn
from app.main import app

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
```

**Step 2: Test that the server starts**

Run: `python run.py` (then Ctrl+C to stop)
Expected: Server starts on http://0.0.0.0:8000

**Step 3: Commit**

```bash
git add run.py
git commit -m "feat: add application entry point"
```

---

## Task 18: Knowledge Base Ingestion Script

**Files:**
- Create: `scripts/ingest_kb.py`

**Step 1: Create ingestion script**

Create `scripts/ingest_kb.py`:

```python
#!/usr/bin/env python
"""
Script to ingest Q&A documents into the vector store.
"""
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
```

**Step 2: Make script executable**

Run: `chmod +x scripts/ingest_kb.py`

**Step 3: Commit**

```bash
git add scripts/
git commit -m "feat: add knowledge base ingestion script"
```

---

## Task 19: README Documentation

**Files:**
- Create: `README.md`

**Step 1: Create comprehensive README**

Create `README.md`:

```markdown
# Telco Customer Service AI Agent

AI-powered customer service agent for a telecommunications company, built with FastAPI, LangChain, OpenAI GPT-4o, Qdrant, and Langfuse.

## Features

- 🤖 RAG-powered chatbot using Q&A knowledge base
- 🔍 Semantic search with Qdrant Cloud vector store
- 📊 Prompt management and tracing with Langfuse
- 🚀 Built with LangChain v1.2.11 and `create_agent`
- 🔄 Automatic escalation to human agents when needed

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry
- OpenAI API key
- Langfuse account
- Qdrant Cloud account

### Installation

```bash
# Install dependencies
poetry install

# Copy environment template
cp .env.example .env

# Edit .env with your credentials
# OPENAI_API_KEY=...
# LANGFUSE_PUBLIC_KEY=...
# LANGFUSE_SECRET_KEY=...
# QDRANT_URL=...
# QDRANT_API_KEY=...
```

### Knowledge Base Setup

```bash
# Ingest Q&A documents into Qdrant
poetry run python scripts/ingest_kb.py
```

### Create System Prompt in Langfuse

Create a chat prompt in Langfuse with the name `telco-customer-service-agent`:

```yaml
type: chat
name: telco-customer-service-agent
prompt:
  - role: system
    content: |
      You are a customer service agent for {{company_name}}, a telecommunications provider.

      Your capabilities:
      - Answer questions about billing, service plans, and troubleshooting
      - Use ONLY information from the retrieved knowledge base
      - If the information is not in the knowledge base, acknowledge it honestly

      Escalation rules:
      - Escalate to human if you cannot confidently answer from the knowledge base
      - DO NOT make up or hallucinate information
      - Clearly state when you don't know something

      Tone: Professional, helpful, concise

      When to escalate:
      - No relevant information found in knowledge base
      - Customer requests to speak to human
      - Complex billing disputes requiring manual review

      Escalation contact: {{escalation_contact}}

variables:
  company_name:
    type: string
    default: "MyTelco"
  escalation_contact:
    type: string
    default: "call 123 or use the MyTelco app"

labels:
  - production
```

### Run the Application

```bash
# Development server
poetry run python run.py

# Or using uvicorn directly
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Usage

### Chat Endpoint

```bash
curl -X POST "http://localhost:8000/chat" \\
  -H "Content-Type: application/json" \\
  -d '{
    "message": "What are your service plans?",
    "conversation_history": []
  }'
```

Response:
```json
{
  "reply": "We offer three plans...",
  "escalate": false,
  "sources": ["service_plans.md"],
  "confidence_score": 0.85
}
```

## System Prompt Design

The system prompt is structured with:

1. **Clear role definition** - Sets boundaries as a Telco customer service agent
2. **Explicit capabilities** - Lists what the agent can and cannot do
3. **Escalation rules** - Defines when to escalate to prevent hallucination
4. **Tone guidance** - Ensures consistent customer experience
5. **Variables** - `company_name` and `escalation_contact` for flexibility

## Chunking Strategy

Instead of using `RecursiveCharacterTextSplitter`, we use **Q&A pairs** as our chunking strategy:

### Why Q&A Pairs?

- **Better semantic matching**: Customer queries are naturally question-based
- **Self-contained chunks**: Each Q&A pair is complete, no context needed
- **No broken context**: No issues with chunk boundaries
- **Easier maintenance**: Can be created and validated manually

### Knowledge Base Structure

Each Q&A document has:
- `question`: Natural language query
- `answer`: Complete response
- `source`: Document name
- `category`: billing, plans, or troubleshooting

## Embedding Model

**Selected**: `text-embedding-3-small` with 512 dimensions

**Reasoning**:
- Cost-effective for this use case
- Fast response times
- Sufficient quality for document-based Q&A
- Smaller vectors = faster similarity search in Qdrant

## Limitations & Production Improvements

### Current Limitations

1. **Manual Q&A creation**: Documents are manually converted to Q&A pairs
2. **Simple retrieval**: Basic similarity search without re-ranking
3. **Single collection**: All documents in one Qdrant collection

### Production Improvements

1. **Automatic Q&A generation**: Use LLM to generate Q&A pairs from documents
2. **Hybrid search**: Combine semantic + keyword search
3. **Re-ranking**: Use cross-encoder for better relevance
4. **Multi-tenancy**: Separate collections per customer/organization
5. **Query caching**: Cache frequent queries with Langfuse
6. **A/B testing**: Test different prompt versions

## Project Structure

```
telco/
├── app/
│   ├── main.py              # FastAPI application
│   ├── api/
│   │   └── routes/
│   │       └── chat.py      # /chat endpoint
│   ├── services/
│   │   ├── llm/
│   │   │   └── agent.py     # LangChain agent
│   │   └── rag/
│   │       ├── retriever.py # RAG retrieval
│   │       └── knowledge_base.py
│   ├── core/
│   │   └── config.py        # Configuration
│   └── prompts/
│       └── langfuse.py      # Langfuse integration
├── data/kb/                 # Knowledge base Q&A files
├── scripts/
│   └── ingest_kb.py        # Ingestion script
├── tests/                   # Test suite
├── .env.example
├── pyproject.toml
└── README.md
```

## Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=app --cov-report=html

# Run specific test file
poetry run pytest tests/test_chat_endpoint.py -v
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API Framework | FastAPI 0.135.1 |
| Agent Framework | LangChain 1.2.11 |
| LLM | OpenAI GPT-4o |
| Vector Store | Qdrant Cloud |
| Embeddings | OpenAI text-embedding-3-small |
| Prompt Management | Langfuse 4.0.0 |
| Tracing | Langfuse CallbackHandler |

## License

MIT License - Built as a technical assignment for AI Engineer position.
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add comprehensive README"
```

---

## Task 20: Final Integration Test

**Step 1: Run all tests**

Run: `poetry run pytest -v`
Expected: All tests pass

**Step 2: Test API manually**

```bash
# Start server
poetry run python run.py

# In another terminal, test the endpoint
curl -X POST "http://localhost:8000/chat" \\
  -H "Content-Type: application/json" \\
  -d '{
    "message": "What are your service plans?",
    "conversation_history": []
  }'
```

Expected: Successful response with plan information

**Step 3: Verify Langfuse tracing**

1. Go to Langfuse dashboard
2. Check that traces are being logged
3. Verify prompt management is working

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: final integration test fixes"
```

---

## Task 21: Create Langfuse System Prompt

**Manual Step - In Langfuse UI**

1. Login to Langfuse Cloud
2. Go to Prompt Management
3. Create new prompt:
   - **Name**: `telco-customer-service-agent`
   - **Type**: Chat
   - **Label**: production

4. Use this prompt content:

```yaml
type: chat
name: telco-customer-service-agent
prompt:
  - role: system
    content: |
      You are a customer service agent for {{company_name}}, a telecommunications provider.

      Your capabilities:
      - Answer questions about billing, service plans, and troubleshooting
      - Use ONLY information from the retrieved knowledge base
      - If the information is not in the knowledge base, acknowledge it honestly

      Escalation rules:
      - Escalate to human if you cannot confidently answer from the knowledge base
      - DO NOT make up or hallucinate information
      - Clearly state when you don't know something

      Tone: Professional, helpful, concise

      When to escalate:
      - No relevant information found in knowledge base
      - Customer requests to speak to human
      - Complex billing disputes requiring manual review

      Escalation contact: {{escalation_contact}}

variables:
  company_name:
    type: string
    default: "MyTelco"
  escalation_contact:
    type: string
    default: "call 123 or use the MyTelco app"

labels:
  - production
```

---

## Task 22: Final Verification

**Step 1: Verify all requirements are met**

Check against assignment requirements:

- ✅ `/chat` endpoint with FastAPI
- ✅ Accept user message and conversation history
- ✅ Use OpenAI GPT-4o
- ✅ System prompt from Langfuse
- ✅ Return JSON with reply and escalate flag
- ✅ Set `escalate: true` when cannot answer
- ✅ RAG pipeline with Qdrant
- ✅ Knowledge base ingestion
- ✅ .env.example included
- ✅ README with explanations

**Step 2: Final commit**

```bash
git add -A
git commit -m "feat: complete Telco AI Agent implementation"
```

**Step 3: Create git tag (optional)**

```bash
git tag -a v0.1.0 -m "Question 1 implementation complete"
```

---

## Summary

This implementation plan builds a complete Customer Service AI Agent for a Telco company with:

1. **Modular architecture** - Ready for Q2 expansion
2. **LangChain v1.2.11** - Using `create_agent`
3. **Q&A pair chunking** - Better than naive text splitting
4. **Langfuse integration** - Prompt management and tracing
5. **Qdrant Cloud** - Persistent vector storage
6. **Comprehensive tests** - Unit, integration, and endpoint tests

**Total estimated time**: 4-6 hours for implementation
