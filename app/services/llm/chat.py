import asyncio
import json
import re
from collections.abc import AsyncIterator
from pydantic import BaseModel
from typing import Optional
from fastapi import WebSocket
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

    def chat(self, message: str, conversation_history: list[dict], session_id: Optional[str] = None) -> ChatResponse:
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
                "langfuse_session_id": session_id or "default",
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

    async def chat_stream(self, message: str, conversation_history: list[dict], session_id: Optional[str] = None) -> AsyncIterator[str]:
        """Stream chat response via Server-Sent Events.

        Yields SSE-formatted strings:
        - Token events: "data: {"type": "token", "content": "..."}\n\n"
        - End event: "data: {"type": "end", "reply": "...", "confidence_score": 0.8, "escalate": false, "sources": [...]}\n\n"
        """
        # Prepare messages (same as chat method)
        messages = conversation_history.copy()
        messages.append({"role": "user", "content": message})

        lc_messages = []
        for msg in messages:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            else:
                lc_messages.append(AIMessage(content=msg["content"]))

        config = {
            "callbacks": [self.handler.handler],
            "metadata": {
                "langfuse_session_id": session_id or "default",
                "langfuse_tags": ["chat", "telco-agent", "stream"]
            }
        }

        full_reply = ""

        # Stream tokens from agent
        async for chunk in self.agent.astream({"messages": lc_messages}, config):
            # chunk is a dict with 'data' key containing (AIMessageChunk, metadata) tuple
            data = chunk.get("data")
            if data and len(data) >= 1:
                message_chunk = data[0]
                if hasattr(message_chunk, "content") and message_chunk.content:
                    full_reply += message_chunk.content
                    yield f"data: {json.dumps({'type': 'token', 'content': message_chunk.content})}\n\n"

        # Extract sources from the final result (need to get full result)
        # For now, we'll re-invoke to get the full result with tool outputs
        # This is not ideal but works for the simple case
        result = self.agent.invoke({"messages": lc_messages}, config)
        sources = self._extract_sources(result)

        # Determine escalate based on keywords in reply
        escalate_keywords = ["human agent", "speak to human", "representative", "escalate"]
        escalate = any(keyword.lower() in full_reply.lower() for keyword in escalate_keywords)

        # Heuristic confidence score
        confidence_score = 0.8 if sources else 0.5

        # Send final end event
        end_event = {
            "type": "end",
            "reply": full_reply,
            "confidence_score": confidence_score,
            "escalate": escalate,
            "sources": sources
        }
        yield f"data: {json.dumps(end_event)}\n\n"

    async def chat_websocket(self, websocket: WebSocket, session_id: Optional[str] = None) -> None:
        """Handle WebSocket chat communication.

        Receives messages via websocket.receive_json():
        - {"type": "message", "message": "...", "conversation_history": [...]}
        - {"type": "cancel"}

        Sends responses via websocket.send_json():
        - {"type": "token", "content": "..."}
        - {"type": "end", "reply": "...", "confidence_score": 0.8, "escalate": false, "sources": [...]}
        - {"type": "error", "message": "..."}
        """
        try:
            # Receive initial message
            data = await websocket.receive_json()

            if data.get("type") == "cancel":
                await websocket.send_json({"type": "end", "reply": "", "cancelled": True})
                return

            if data.get("type") != "message":
                await websocket.send_json({"type": "error", "message": "Expected message type"})
                return

            message = data.get("message", "")
            conversation_history = data.get("conversation_history", [])
            session_id = data.get("session_id") or session_id

            # Prepare messages
            messages = conversation_history.copy()
            messages.append({"role": "user", "content": message})

            lc_messages = []
            for msg in messages:
                if msg["role"] == "user":
                    lc_messages.append(HumanMessage(content=msg["content"]))
                else:
                    lc_messages.append(AIMessage(content=msg["content"]))

            config = {
                "callbacks": [self.handler.handler],
                "metadata": {
                    "langfuse_session_id": session_id or "default",
                    "langfuse_tags": ["chat", "telco-agent", "websocket"]
                }
            }

            full_reply = ""

            # Stream tokens
            async for chunk in self.agent.astream({"messages": lc_messages}, config):
                # Check for cancel
                try:
                    cancel_msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.01)
                    if cancel_msg.get("type") == "cancel":
                        await websocket.send_json({"type": "end", "reply": full_reply, "cancelled": True})
                        return
                except asyncio.TimeoutError:
                    pass

                data = chunk.get("data")
                if data and len(data) >= 1:
                    message_chunk = data[0]
                    if hasattr(message_chunk, "content") and message_chunk.content:
                        full_reply += message_chunk.content
                        await websocket.send_json({"type": "token", "content": message_chunk.content})

            # Get final result for sources
            result = self.agent.invoke({"messages": lc_messages}, config)
            sources = self._extract_sources(result)

            # Determine escalate
            escalate_keywords = ["human agent", "speak to human", "representative", "escalate"]
            escalate = any(keyword.lower() in full_reply.lower() for keyword in escalate_keywords)
            confidence_score = 0.8 if sources else 0.5

            await websocket.send_json({
                "type": "end",
                "reply": full_reply,
                "confidence_score": confidence_score,
                "escalate": escalate,
                "sources": sources
            })

        except Exception as e:
            await websocket.send_json({"type": "error", "message": str(e)})

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
