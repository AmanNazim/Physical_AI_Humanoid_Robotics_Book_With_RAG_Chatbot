"""
Embedding request/response schemas for the RAG Chatbot API.
"""
from typing import Dict, Any, Optional
from pydantic import BaseModel


class EmbeddingRequest(BaseModel):
    """Request schema for embedding operations."""
    text: str
    document_metadata: Optional[Dict[str, Any]] = None
    document_id: Optional[str] = None
    title: Optional[str] = None
    source: Optional[str] = "manual"


class EmbeddingResponse(BaseModel):
    """Response schema for embedding operations."""
    status: str  # "success" or "error"
    message: str
    chunks_processed: Optional[int] = None
    embeddings_generated: Optional[int] = None
    document_id: Optional[str] = None
    elapsed_ms: Optional[float] = None