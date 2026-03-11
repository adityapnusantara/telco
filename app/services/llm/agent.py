from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from app.prompts.langfuse import get_system_prompt
from app.services.rag.retriever import get_retriever_tool

_agent = None

def get_agent():
    """Get or create the LangChain agent singleton"""
    global _agent
    if _agent is None:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        system_prompt = get_system_prompt()
        prompt_text = "\n".join([msg["content"] for msg in system_prompt])
        retriever_tool = get_retriever_tool()

        _agent = create_agent(
            model=llm,
            tools=[retriever_tool],
            system_prompt=prompt_text,
        )

    return _agent
