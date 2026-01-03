from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from enum import Enum


class ErrorCode(str, Enum):
    """Enumeration of error codes"""
    VALIDATION_ERROR = "VALIDATION_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    EMBEDDING_ERROR = "EMBEDDING_ERROR"
    SEARCH_ERROR = "SEARCH_ERROR"
    LLM_ERROR = "LLM_ERROR"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    AUTHORIZATION_ERROR = "AUTHORIZATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"


class ErrorDetail(BaseModel):
    """Model for error details"""
    code: ErrorCode = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Timestamp of the error")


class ErrorResponse(BaseModel):
    """Model for error responses"""
    error: ErrorDetail = Field(..., description="Error details")