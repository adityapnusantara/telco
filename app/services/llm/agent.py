from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from app.prompts.langfuse import get_system_prompt
from app.services.rag.retriever import RetrieverTool
from app.services.rag.vector_store import VectorStore


class StructuredChatResponse(BaseModel):
    """Structured response schema for agent output"""
    reply: str = Field(description="The natural language response to the user's question")
    confidence_score: float = Field(
        description="Confidence score from 0.0 to 1.0 indicating how certain the agent is about its answer",
        ge=0.0,
        le=1.0
    )
    escalate: bool = Field(description="True if the user should be escalated to a human agent")


class Agent:
    """LangChain agent wrapper with explicit initialization and DI"""

    def __init__(self, vector_store: VectorStore, retriever_tool=None):
        self._vector_store = vector_store
        self._llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self._system_prompt = get_system_prompt()

        if retriever_tool:
            self._retriever_tool = retriever_tool.tool
        else:
            # Create RetrieverTool if not provided
            self._retriever_tool = RetrieverTool(vector_store).tool

        self._agent = create_agent(
            model=self._llm,
            tools=[self._retriever_tool],
            system_prompt=self._system_prompt,  # Langfuse returns compiled string directly
            response_format=StructuredChatResponse
        )

    def invoke(self, messages, config):
        """Invoke the agent with messages and config"""
        return self._agent.invoke(messages, config)
