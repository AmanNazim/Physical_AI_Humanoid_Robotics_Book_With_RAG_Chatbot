from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class ChatMode(str, Enum):
    """Enumeration of chat modes"""
    FULL_BOOK = "full_book"
    SELECTED_TEXT_ONLY = "selected_text_only"


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    query: str = Field(..., description="User query", min_length=1, max_length=2000)
    mode: ChatMode = Field(default=ChatMode.FULL_BOOK, description="Retrieval mode")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top results to retrieve")
    selected_text: Optional[str] = Field(None, description="User-provided text for selected-text-only mode")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    session_id: Optional[str] = Field(None, description="Optional session identifier")


class Citation(BaseModel):
    """Model for citations in chat responses"""
    chunk_id: str
    text: str
    document_reference: str
    page_reference: Optional[int] = None
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    success: bool = True
    answer: str = Field(..., description="Generated answer to the query")
    citations: List[Citation] = Field(default_factory=list, description="List of citations for the answer")
    query: str = Field(..., description="Original user query")
    mode: ChatMode = Field(..., description="Retrieval mode used")
    retrieved_chunks_count: int = Field(..., description="Number of chunks retrieved")
    processing_time: float = Field(..., description="Time taken to process the query in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of response")


class ChatHistoryItem(BaseModel):
    """Model for chat history items"""
    query: str
    answer: str
    citations: List[Citation]
    timestamp: datetime
    mode: ChatMode


class ChatHistoryResponse(BaseModel):
    """Response model for chat history"""
    success: bool = True
    history: List[ChatHistoryItem]
    user_id: Optional[str] = None
    total_count: int


class ChatConfig(BaseModel):
    """Configuration for chat behavior"""
    default_top_k: int = Field(default=5, ge=1, le=20, description="Default number of results to retrieve")
    max_query_length: int = Field(default=2000, ge=100, le=10000, description="Maximum query length allowed")
    enable_citations: bool = Field(default=True, description="Whether to include citations in responses")
    enable_history: bool = Field(default=True, description="Whether to maintain chat history")
    default_mode: ChatMode = Field(default=ChatMode.FULL_BOOK, description="Default retrieval mode")