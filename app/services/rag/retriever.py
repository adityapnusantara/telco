from langchain_core.tools import tool
from app.services.rag.vector_store import VectorStore


class RetrieverTool:
    """Wrapper for LangChain retriever tool with DI"""

    def __init__(self, vector_store: VectorStore):
        self._vector_store = vector_store
        self._tool = self._create_tool()

    def _create_tool(self):
        """Create the LangChain tool"""
        @tool
        def search_knowledge_base(query: str) -> str:
            """Search the Telco knowledge base for relevant information"""
            retriever = self._vector_store.store.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(query)

            if not docs:
                return "No relevant information found in the knowledge base."

            results = []
            for doc in docs:
                source = doc.metadata.get("source", "unknown")
                category = doc.metadata.get("category", "general")
                results.append(f"[{category} - {source}]: {doc.page_content}")

            return "\n\n".join(results)

        return search_knowledge_base

    @property
    def tool(self):
        """Get the LangChain tool"""
        return self._tool
