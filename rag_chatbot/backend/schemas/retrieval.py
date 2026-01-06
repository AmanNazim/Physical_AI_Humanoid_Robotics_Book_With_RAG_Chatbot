"""
Retrieval request/response schemas for the RAG Chatbot API.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class Source(BaseModel):
    """Schema for retrieved source documents."""
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class RetrievalRequest(BaseModel):
    """Request schema for retrieval operations."""
    query: str
    top_k: int = 5
    filters: Optional[Dict[str, Any]] = None
    min_score: Optional[float] = 0.0


class RetrievalResponse(BaseModel):
    """Response schema for retrieval operations."""
    sources: List[Source]
    query: str
    retrieved_count: int