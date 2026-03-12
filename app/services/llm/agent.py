import warnings
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from app.prompts.langfuse import get_system_prompt
from app.services.rag.retriever import get_retriever_tool
from app.services.rag.vector_store import VectorStore


class Agent:
    """LangChain agent wrapper with explicit initialization and DI"""

    def __init__(self, vector_store: VectorStore):
        self._vector_store = vector_store
        self._llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self._system_prompt = get_system_prompt()
        prompt_text = "\n".join([msg["content"] for msg in self._system_prompt])
        self._retriever_tool = get_retriever_tool()
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
