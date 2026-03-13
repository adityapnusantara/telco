from collections.abc import AsyncIterator
from typing import Any

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from app.prompts.langfuse import get_agent_prompt
from app.services.rag.retriever import RetrieverTool
from app.services.rag.vector_store import VectorStore


class Agent:
    """LangChain agent wrapper with explicit initialization and DI"""

    def __init__(self, vector_store: VectorStore, retriever_tool=None):
        self._vector_store = vector_store
        self._prompt = get_agent_prompt()
        self._retriever_tool = retriever_tool.tool

        self._llm = ChatOpenAI(
            model=self._prompt["model_config"]["model"],
            temperature=self._prompt["model_config"]["temperature"]
        )

        self._agent = create_agent(
            model=self._llm,
            tools=[self._retriever_tool],
            system_prompt=self._prompt["system_prompt"]  # Langfuse returns compiled string directly
            # No response_format - natural text output only
        )

    def invoke(self, messages: dict[str, Any], config: dict[str, Any]) -> Any:
        """Invoke the agent with messages and config"""
        return self._agent.invoke(messages, config)

    async def astream(self, messages: dict[str, Any], config: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Stream agent responses token-by-token.

        Yields chunks from LangChain's astream with stream_mode="messages".
        """
        async for chunk in self._agent.astream(
            messages,
            config,
            stream_mode="messages",
            version="v2"
        ):
            yield chunk
