from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from .request_models import DocumentMetadata


class Source(BaseModel):
    """Model for source information in responses"""
    chunk_id: str = Field(..., description="ID of the source chunk")
    document_id: str = Field(..., description="ID of the source document")
    text: str = Field(..., description="Text content of the source chunk")
    score: float = Field(..., description="Similarity score of the source chunk")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class BaseResponse(BaseModel):
    """Base response model with common fields"""
    status: str = Field(default="success", description="Status of the response")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the response")


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    answer: str = Field(..., description="The generated answer")
    sources: List[Source] = Field(default=[], description="List of sources used in the answer")
    latency_ms: Optional[float] = Field(None, description="Processing latency in milliseconds")
    session_id: Optional[str] = Field(None, description="Session ID for the conversation")


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    answer: str = Field(..., description="The generated answer")
    sources: List[Source] = Field(default=[], description="List of sources used in the answer")
    latency_ms: Optional[float] = Field(None, description="Processing latency in milliseconds")


class IngestionResponse(BaseModel):
    """Response model for ingestion endpoint"""
    status: str = Field(default="success", description="Status of the ingestion")
    document_id: str = Field(..., description="ID of the ingested document")
    chunks_created: int = Field(..., description="Number of chunks created during ingestion")
    vectors_stored: int = Field(..., description="Number of vectors stored in the database")
    elapsed_ms: Optional[float] = Field(None, description="Processing time in milliseconds")


class SearchResponse(BaseModel):
    """Response model for search endpoint"""
    results: List[Source] = Field(..., description="List of search results")
    query: str = Field(..., description="Original search query")
    elapsed_ms: Optional[float] = Field(None, description="Search processing time in milliseconds")


class DocumentResponse(BaseModel):
    """Response model for document endpoint"""
    documents: List[DocumentMetadata] = Field(..., description="List of document metadata")


class HealthResponse(BaseModel):
    """Response model for health endpoint"""
    status: str = Field(default="ok", description="Overall system status")
    version: str = Field(default="v1", description="API version")
    qdrant_connected: bool = Field(..., description="Whether Qdrant is connected")
    postgres_connected: bool = Field(..., description="Whether Postgres is connected")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")


class ConfigResponse(BaseModel):
    """Response model for config endpoint"""
    version: str = Field(default="v1", description="API version")
    streaming_enabled: bool = Field(default=True, description="Whether streaming is enabled")
    max_context_chunks: int = Field(..., description="Maximum number of context chunks allowed")
    features: Dict[str, Any] = Field(default={}, description="Available features and their status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Config retrieval timestamp")