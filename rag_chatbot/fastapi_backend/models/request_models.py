from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    query: str = Field(..., description="User query to process")
    session_id: Optional[str] = Field(None, description="Session ID for conversation context")
    max_context: int = Field(default=5, ge=1, le=20, description="Maximum number of context chunks to retrieve")
    stream: bool = Field(default=False, description="Whether to stream the response")


class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str = Field(..., description="User query to process")
    max_context: int = Field(default=5, ge=1, le=20, description="Maximum number of context chunks to retrieve")


class IngestTextRequest(BaseModel):
    """Request model for text ingestion endpoint"""
    text: str = Field(..., description="Text content to ingest", min_length=1)
    title: str = Field(..., description="Title of the document", min_length=1)
    source: str = Field(default="manual", description="Source type (manual, pdf, md)")
    document_id: Optional[str] = Field(None, description="Optional document ID (auto-generated if not provided)")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata for the document")


class DocumentMetadata(BaseModel):
    """Model for document metadata"""
    document_id: str = Field(..., description="Unique identifier for the document")
    title: str = Field(..., description="Title of the document", min_length=1)
    source: str = Field(..., description="Source type (manual, pdf, md)")
    chunk_count: int = Field(..., ge=0, description="Number of chunks in the document")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")


class EmbeddingRequest(BaseModel):
    """Request model for embedding endpoint"""
    text: str = Field(..., description="Text to generate embeddings for", min_length=1)
    task_type: str = Field(default="SEMANTIC_SIMILARITY", description="Type of embedding task")
    output_dimensionality: Optional[int] = Field(None, description="Output dimensionality for embeddings")


class SearchRequest(BaseModel):
    """Request model for search endpoint"""
    query: str = Field(..., description="Search query", min_length=1)
    top_k: int = Field(default=5, ge=1, le=50, description="Number of top results to retrieve")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Additional filters for search")


class SemanticSearchRequest(SearchRequest):
    """Request model for semantic search endpoint"""
    pass


class HybridSearchRequest(SearchRequest):
    """Request model for hybrid search endpoint"""
    keyword_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Weight for keyword search")
    semantic_weight: float = Field(default=0.7, ge=0.0, le=1.0, description="Weight for semantic search")


class ConversationStateRequest(BaseModel):
    """Request model for conversation state endpoint"""
    session_id: str = Field(..., description="Session ID for the conversation")
    user_id: Optional[str] = Field(None, description="User ID")
    state_data: Dict[str, Any] = Field(..., description="State data to store")