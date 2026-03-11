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


def chat(message: str, conversation_history: list[dict], conversation_id: Optional[str] = None) -> ChatResponse:
    """Process a chat message using the RAG agent"""
    agent = get_agent()
    handler = get_langfuse_handler()

    messages = conversation_history.copy()
    messages.append({"role": "user", "content": message})

    config = {
        "callbacks": [handler],
        "metadata": {
            "langfuse_session_id": conversation_id or "default",
            "langfuse_tags": ["chat", "telco-agent"]
        }
    }

    result = agent.invoke({"messages": messages}, config=config)
    reply = result.messages[-1]["content"]

    escalate = _should_escalate(reply)

    return ChatResponse(reply=reply, escalate=escalate)


def _should_escalate(reply: str) -> bool:
    """Determine if escalation is needed"""
    escalation_indicators = [
        "cannot help", "don't know", "unable to assist",
        "speak to a human", "transfer to agent", "escalate"
    ]
    reply_lower = reply.lower()
    return any(indicator in reply_lower for indicator in escalation_indicators)
