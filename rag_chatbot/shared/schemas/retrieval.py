from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from .chat import Citation


class RetrievalRequest(BaseModel):
    """Request model for retrieval endpoint"""
    query: str = Field(..., description="Query for vector search", min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top results to retrieve")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional filters for search")
    document_reference: Optional[str] = Field(None, description="Optional document reference to limit search")


class RetrievalResult(BaseModel):
    """Model for individual retrieval result"""
    chunk_id: str
    text: str
    document_reference: str
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    page_reference: Optional[int] = None
    section_title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class RetrievalResponse(BaseModel):
    """Response model for retrieval endpoint"""
    success: bool = True
    query: str
    results: List[RetrievalResult]
    top_k: int
    processing_time: float = Field(..., description="Time taken to process the query in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    total_results: int


class BatchRetrievalRequest(BaseModel):
    """Request model for batch retrieval"""
    queries: List[str] = Field(..., min_items=1, max_items=50, description="List of queries")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top results per query")


class BatchRetrievalResponse(BaseModel):
    """Response model for batch retrieval"""
    success: bool = True
    requests_count: int
    results: List[RetrievalResponse]
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EmbedRequest(BaseModel):
    """Request model for embedding endpoint"""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="Texts to embed")
    input_type: str = Field(default="search_document", description="Type of input for embeddings")


class EmbedResponse(BaseModel):
    """Response model for embedding endpoint"""
    success: bool = True
    embeddings: List[List[float]] = Field(..., description="Generated embeddings")
    texts_count: int
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DocumentUploadRequest(BaseModel):
    """Request model for document upload and processing"""
    document_content: str = Field(..., description="Content of the document to process")
    document_reference: str = Field(..., description="Reference for the document")
    chunk_size: int = Field(default=1000, ge=100, le=2000, description="Size of text chunks")
    overlap: int = Field(default=100, ge=0, le=500, description="Overlap between chunks")


class DocumentUploadResponse(BaseModel):
    """Response model for document upload"""
    success: bool = True
    document_reference: str
    chunks_processed: int
    embeddings_generated: int
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    chunk_ids: List[str]