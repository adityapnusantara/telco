from pydantic import BaseModel
from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage
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

        # Convert dict messages to LangChain Message objects
        lc_messages = []
        for msg in messages:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            else:
                lc_messages.append(AIMessage(content=msg["content"]))

        config = {
            "callbacks": [self.handler.handler],
            "metadata": {
                "langfuse_session_id": conversation_id or "default",
                "langfuse_tags": ["chat", "telco-agent"]
            }
        }

        result = self.agent.invoke({"messages": lc_messages}, config=config)

        # Extract reply from result (dict with "messages" key)
        messages_list = result["messages"]
        last_message = messages_list[-1]
        reply = last_message.content

        escalate = self._should_escalate(reply)

        return ChatResponse(
            reply=reply,
            escalate=escalate
        )

    def _should_escalate(self, reply: str) -> bool:
        """Determine if escalation is needed"""
        escalation_indicators = [
            "cannot help", "don't know", "unable to assist",
            "speak to a human", "transfer to agent", "escalate"
        ]
        reply_lower = reply.lower()
        return any(indicator in reply_lower for indicator in escalation_indicators)
