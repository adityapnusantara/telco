from pydantic import BaseModel, Field
from typing import Optional, List

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., min_length=1, description="User's message")
    session_id: Optional[str] = Field(None, description="Session ID for tracking conversation")
    conversation_history: Optional[List[dict]] = Field(default_factory=list, description="Previous messages")

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    reply: str = Field(..., description="Agent's response")
    escalate: bool = Field(..., description="Whether to escalate to human")
    sources: Optional[List[str]] = Field(None, description="Retrieved document names")
    confidence_score: Optional[float] = Field(None, description="Confidence score (0-1)")
