import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator
from pydantic import BaseModel, Field
from typing import Optional
from fastapi import WebSocket
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from app.prompts.langfuse import get_system_prompt, get_model_config, get_classification_prompt_obj, get_classification_config
from .agent import Agent
from .callbacks import CallbackHandler

logger = logging.getLogger(__name__)


class ReplyClassification(BaseModel):
    """Classification result for reply metadata"""
    confidence_score: float = Field(description="Confidence 0.0-1.0", ge=0.0, le=1.0)
    escalate: bool = Field(description="True if should escalate to human")


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

        # Classification Agent - create_agent with empty tools list
        self._classification_prompt_obj = get_classification_prompt_obj()
        model_config = get_classification_config()

        classification_llm = ChatOpenAI(
            model=model_config["model"],
            temperature=model_config["temperature"]
        )

        # System prompt for classification agent (static, no variables)
        classification_system_prompt = (
            "You are a customer service response classifier. "
            "Analyze replies and provide metadata about confidence and escalation needs. "
            "Always respond with valid JSON matching the required schema."
        )

        self._classification_agent = create_agent(
            model=classification_llm,
            tools=[],  # Empty - no tools needed for classification
            system_prompt=classification_system_prompt,
            response_format=ReplyClassification
        )

    async def _classify_reply(self, reply: str, has_sources: bool) -> ReplyClassification:
        """Classify reply to extract confidence_score and escalate using LLM.

        Args:
            reply: The agent's natural language response
            has_sources: Whether sources were found in the knowledge base

        Returns:
            ReplyClassification with confidence_score and escalate
        """
        # Compile prompt with variables
        context = 'Sources found from knowledge base' if has_sources else 'No sources available from knowledge base'
        compiled_prompt = self._classification_prompt_obj.compile(
            reply=reply,
            context=context
        )

        result = self._classification_agent.invoke(
            {"messages": [{"role": "user", "content": compiled_prompt}]},
            config={
                "callbacks": [self.handler.handler],
                "metadata": {
                    "langfuse_tags": ["classification", "metadata-extraction"],
                    "classification_task": "reply_metadata"
                }
            }
        )

        return result["structured_response"]

    async def chat(self, message: str, conversation_history: list[dict], session_id: Optional[str] = None) -> ChatResponse:
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

        # Extract reply from last message (natural text, not JSON)
        messages_list = result.get("messages", [])

        # Handle empty messages list
        if not messages_list:
            return ChatResponse(
                reply="I apologize, but I couldn't generate a response. Please try again.",
                escalate=True,
                confidence_score=0.0,
                sources=None
            )
        last_message = messages_list[-1]
        reply = last_message.content

        # Extract sources from tool results
        sources = self._extract_sources(result)

        # Classify metadata using LLM
        classification = await self._classify_reply(reply, has_sources=bool(sources))

        return ChatResponse(
            reply=reply,
            escalate=classification.escalate,
            confidence_score=classification.confidence_score,
            sources=sources
        )

    async def chat_stream(self, message: str, conversation_history: list[dict], session_id: Optional[str] = None) -> AsyncIterator[str]:
        """Stream chat response via Server-Sent Events.

        Yields SSE-formatted strings:
        - Token events: "data: {"type": "token", "content": "..."}\n\n"
        - End event: "data: {"type": "end", "reply": "...", "confidence_score": 0.8, "escalate": false, "sources": [...]}\n\n"
        - Error event: "data: {"type": "error", "message": "..."}\n\n"
        """
        try:
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

            # Stream tokens from agent (only AIMessage, skip ToolMessage)
            async for chunk in self.agent.astream({"messages": lc_messages}, config):
                # chunk is a dict with 'data' key containing (AIMessageChunk, metadata) tuple
                data = chunk.get("data")
                if data and len(data) >= 1:
                    message_chunk = data[0]
                    # Only stream AIMessage content, skip ToolMessage (tool results)
                    if message_chunk.content:
                        # Check if this is an AIMessage (not ToolMessage)
                        # ToolMessage has 'tool_call_id' attribute, AIMessage doesn't
                        if not hasattr(message_chunk, 'tool_call_id'):
                            full_reply += message_chunk.content
                            yield f"data: {json.dumps({'type': 'token', 'content': message_chunk.content})}\n\n"

            # Extract sources from the final result (need to get full result)
            # For now, we'll re-invoke to get the full result with tool outputs
            # This is not ideal but works for the simple case
            result = self.agent.invoke({"messages": lc_messages}, config)
            sources = self._extract_sources(result)

            # Classify metadata using LLM (replaces keyword heuristics)
            classification = await self._classify_reply(full_reply, has_sources=bool(sources))

            # Send final end event with CLASSIFIED metadata
            end_event = {
                "type": "end",
                "reply": full_reply,
                "confidence_score": classification.confidence_score,
                "escalate": classification.escalate,
                "sources": sources
            }
            yield f"data: {json.dumps(end_event)}\n\n"

        except Exception as e:
            # Log the actual error for debugging
            logger.error(f"Error in chat_stream: {e}", exc_info=True)
            # Send generic error event to client (don't expose raw exceptions)
            error_event = {
                "type": "error",
                "message": "An error occurred while processing your request. Please try again."
            }
            yield f"data: {json.dumps(error_event)}\n\n"

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
                    # Only stream AIMessage content, skip ToolMessage (tool results)
                    if hasattr(message_chunk, "content") and message_chunk.content:
                        # Check if this is an AIMessage (not ToolMessage)
                        # ToolMessage has 'tool_call_id' attribute, AIMessage doesn't
                        if not hasattr(message_chunk, 'tool_call_id'):
                            full_reply += message_chunk.content
                            await websocket.send_json({"type": "token", "content": message_chunk.content})

            # Get final result for sources
            result = self.agent.invoke({"messages": lc_messages}, config)
            sources = self._extract_sources(result)

            # Classify metadata using LLM (replaces keyword heuristics)
            classification = await self._classify_reply(full_reply, has_sources=bool(sources))

            await websocket.send_json({
                "type": "end",
                "reply": full_reply,
                "confidence_score": classification.confidence_score,
                "escalate": classification.escalate,
                "sources": sources
            })

        except Exception as e:
            # Log the actual error for debugging
            logger.error(f"Error in chat_websocket: {e}", exc_info=True)
            # Send generic error event to client (don't expose raw exceptions)
            await websocket.send_json({
                "type": "error",
                "message": "An error occurred while processing your request. Please try again."
            })

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
