import re
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

        # Extract structured response
        structured = result.get("structured_response")
        if structured is None:
            # Fallback handling if structured_response is missing
            return self._fallback_response(result)

        # Extract sources from tool calls
        sources = self._extract_sources(result)

        return ChatResponse(
            reply=structured.reply,
            escalate=structured.escalate,
            confidence_score=structured.confidence_score,
            sources=sources
        )

    def _extract_sources(self, result: dict) -> Optional[list[str]]:
        """Extract source document names from retriever tool calls"""
        sources = []
        messages = result.get("messages", [])

        for msg in messages:
            if hasattr(msg, "tool_calls"):
                for call in msg.tool_calls:
                    if call.get("name") == "search_knowledge_base":
                        # Extract from tool args (query)
                        args = call.get("args", {})
                        # The source names are in the tool results, not args
                        # We need to look at ToolMessage responses
            elif hasattr(msg, "content"):
                # Handle string content (ToolMessage can have string content)
                if isinstance(msg.content, str):
                    # Parse source from formatted response: "[category - source]: content"
                    text = msg.content
                    # Match pattern like "[billing - billing_qna.json]: ..."
                    pattern = r'\[([^\]]+?-([^\]]+?))\]:'
                    matches = re.findall(pattern, text)
                    for match in matches:
                        if len(match) >= 2:
                            source = match[1].strip()  # Strip whitespace from source filename
                            if source not in sources:
                                sources.append(source)
                elif isinstance(msg.content, list):
                    # Tool results come back as content lists
                    for content_item in msg.content:
                        if isinstance(content_item, dict) and "text" in content_item:
                            # Parse source from formatted response: "[category - source]: content"
                            text = content_item["text"]
                            # Match pattern like "[billing - billing_qna.json]: ..."
                            pattern = r'\[([^\]]+?-([^\]]+?))\]:'
                            matches = re.findall(pattern, text)
                            for match in matches:
                                if len(match) >= 2:
                                    source = match[1].strip()  # Strip whitespace from source filename
                                    if source not in sources:
                                        sources.append(source)

        return sources if sources else None

    def _fallback_response(self, result: dict) -> ChatResponse:
        """Fallback response if structured_response is missing"""
        messages_list = result.get("messages", [])

        # Handle empty messages list
        if not messages_list:
            return ChatResponse(
                reply="I apologize, but I couldn't generate a response. Please try again.",
                escalate=True,  # Escalate when we can't generate a response
                confidence_score=None,
                sources=None
            )

        last_message = messages_list[-1]
        reply = last_message.content if hasattr(last_message, 'content') else str(last_message)

        return ChatResponse(
            reply=reply,
            escalate=False,  # Default on fallback
            confidence_score=None,
            sources=self._extract_sources(result)
        )
