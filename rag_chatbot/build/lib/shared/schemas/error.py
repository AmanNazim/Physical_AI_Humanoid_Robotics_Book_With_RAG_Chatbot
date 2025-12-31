from pydantic import BaseModel
from typing import Optional, Dict, Any
from enum import Enum


class ErrorCode(str, Enum):
    """Enumeration of possible error codes"""
    VALIDATION_ERROR = "VALIDATION_ERROR"
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"
    EMBEDDING_ERROR = "EMBEDDING_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    AGENT_ERROR = "AGENT_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"


class ErrorResponse(BaseModel):
    """Standard error response model"""
    success: bool = False
    error_code: ErrorCode
    message: str
    details: Optional[str] = None
    timestamp: str
    request_id: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None

    class Config:
        use_enum_values = True


class ValidationErrorResponse(ErrorResponse):
    """Error response specifically for validation errors"""
    error_code: ErrorCode = ErrorCode.VALIDATION_ERROR
    validation_errors: Optional[Dict[str, str]] = None


class RetrievalErrorResponse(ErrorResponse):
    """Error response specifically for retrieval errors"""
    error_code: ErrorCode = ErrorCode.RETRIEVAL_ERROR
    query: Optional[str] = None


class DatabaseErrorResponse(ErrorResponse):
    """Error response specifically for database errors"""
    error_code: ErrorCode = ErrorCode.DATABASE_ERROR
    operation: Optional[str] = None
    table: Optional[str] = None


class AgentErrorResponse(ErrorResponse):
    """Error response specifically for agent errors"""
    error_code: ErrorCode = ErrorCode.AGENT_ERROR
    query: Optional[str] = None
    context_length: Optional[int] = None