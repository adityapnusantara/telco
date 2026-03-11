from langchain_core.tools import tool
from .vector_store import get_vector_store

@tool
def search_knowledge_base(query: str) -> str:
    """Search the Telco knowledge base for relevant information"""
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information found in the knowledge base."

    results = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        category = doc.metadata.get("category", "general")
        results.append(f"[{category} - {source}]: {doc.page_content}")

    return "\n\n".join(results)

def get_retriever_tool():
    """Get the retriever tool for use with LangChain agents"""
    return search_knowledge_base
