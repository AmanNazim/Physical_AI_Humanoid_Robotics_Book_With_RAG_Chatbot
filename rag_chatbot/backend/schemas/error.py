"""
Error handling schemas for the RAG Chatbot API.
"""
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel


class ErrorCode(str, Enum):
    """Enumeration of possible error codes."""
    # General errors
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"

    # RAG-specific errors
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"
    EMBEDDING_ERROR = "EMBEDDING_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    SUBSYSTEM_ERROR = "SUBSYSTEM_ERROR"


class ErrorDetail(BaseModel):
    """Detailed error information."""
    code: ErrorCode
    message: str
    details: Optional[Any] = None
    timestamp: Optional[str] = None
    request_id: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standardized error response format."""
    error: ErrorDetail
    success: bool = False