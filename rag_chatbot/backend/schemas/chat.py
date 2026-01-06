"""
Chat request/response schemas for the RAG Chatbot API.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from .retrieval import Source


class ChatRequest(BaseModel):
    """Request schema for chat operations."""
    query: str
    max_context: int = 5
    session_id: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000


class ChatResponse(BaseModel):
    """Response schema for chat operations."""
    answer: str
    sources: List[Source]
    session_id: Optional[str] = None
    query: str
    latency_ms: Optional[float] = None


class StreamResponse(BaseModel):
    """Schema for streaming responses."""
    type: str  # "token", "source", "complete", "error"
    content: Optional[str] = None
    sources: Optional[List[Source]] = None
    message: Optional[str] = None
    index: Optional[int] = None
    total_tokens: Optional[int] = None