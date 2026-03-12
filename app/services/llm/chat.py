import warnings
from pydantic import BaseModel
from typing import Optional
from .agent import Agent
from .callbacks import CallbackHandler


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    reply: str
    escalate: bool
    sources: Optional[list[str]] = None
    confidence_score: Optional[float] = None


class ChatService:
    """Chat service with explicit dependency injection"""

    def __init__(self, agent: Agent, handler: CallbackHandler):
        self.agent = agent
        self.handler = handler

    def chat(self, message: str, conversation_history: list[dict], conversation_id: Optional[str] = None) -> ChatResponse:
        """Process a chat message using the RAG agent"""
        messages = conversation_history.copy()
        messages.append({"role": "user", "content": message})

        config = {
            "callbacks": [self.handler.handler],
            "metadata": {
                "langfuse_session_id": conversation_id or "default",
                "langfuse_tags": ["chat", "telco-agent"]
            }
        }

        result = self.agent.invoke({"messages": messages}, config=config)
        reply = result.messages[-1]["content"]

        escalate = self._should_escalate(reply)

        return ChatResponse(reply=reply, escalate=escalate)

    def _should_escalate(self, reply: str) -> bool:
        """Determine if escalation is needed"""
        escalation_indicators = [
            "cannot help", "don't know", "unable to assist",
            "speak to a human", "transfer to agent", "escalate"
        ]
        reply_lower = reply.lower()
        return any(indicator in reply_lower for indicator in escalation_indicators)


# Legacy function for backward compatibility (will be removed after migration)
def chat(message: str, conversation_history: list[dict], conversation_id: Optional[str] = None) -> ChatResponse:
    """Deprecated: Use ChatService class instead"""
    warnings.warn("Use ChatService class instead", DeprecationWarning, stacklevel=2)
    raise NotImplementedError("Use ChatService class instead")


def _should_escalate(reply: str) -> bool:
    """Deprecated: Use ChatService._should_escalate method instead"""
    warnings.warn("Use ChatService._should_escalate method instead", DeprecationWarning, stacklevel=2)
    raise NotImplementedError("Use ChatService class instead")
