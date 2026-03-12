import warnings
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from app.prompts.langfuse import get_system_prompt
from app.services.rag.retriever import RetrieverTool
from app.services.rag.vector_store import VectorStore


class Agent:
    """LangChain agent wrapper with explicit initialization and DI"""

    def __init__(self, vector_store: VectorStore, retriever_tool=None):
        self._vector_store = vector_store
        self._llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self._system_prompt = get_system_prompt()
        prompt_text = "\n".join([msg["content"] for msg in self._system_prompt])

        if retriever_tool:
            self._retriever_tool = retriever_tool.tool
        else:
            # Create RetrieverTool if not provided
            self._retriever_tool = RetrieverTool(vector_store).tool

        self._agent = create_agent(
            model=self._llm,
            tools=[self._retriever_tool],
            system_prompt=prompt_text
        )

    def invoke(self, messages, config):
        """Invoke the agent with messages and config"""
        return self._agent.invoke(messages, config)


# Legacy function for backward compatibility (will be removed after migration)
def get_agent():
    """Deprecated: Use Agent class instead"""
    warnings.warn("Use Agent class instead", DeprecationWarning, stacklevel=2)
    raise NotImplementedError("Use Agent class instead")
