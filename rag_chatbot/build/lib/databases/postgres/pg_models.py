from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class ChunkMetadata(BaseModel):
    """Model for chunk metadata"""
    chunk_id: str
    document_reference: str
    page_reference: Optional[int] = None
    section_title: Optional[str] = None
    chunk_text: str
    embedding_id: str
    processing_version: str
    created_at: str
    updated_at: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LogEntry(BaseModel):
    """Model for log entries"""
    log_id: str
    user_query: str
    retrieved_chunks: List[Dict[str, Any]]
    response: str
    timestamp: str
    retrieval_mode: str  # 'full_book' or 'selected_text'


class ChatHistoryEntry(BaseModel):
    """Model for chat history entries"""
    chat_id: str
    user_id: Optional[str] = None
    query: str
    response: str
    source_chunks: List[Dict[str, Any]]
    timestamp: str


class User(BaseModel):
    """Model for user data"""
    user_id: str
    created_at: str
    email: Optional[str] = None
    profile_metadata: Dict[str, Any] = Field(default_factory=dict)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True
    last_login: Optional[str] = None


class AuditLog(BaseModel):
    """Model for audit log entries"""
    log_id: str
    operation_type: str
    resource_type: str
    resource_id: str
    user_id: Optional[str] = None
    operation_timestamp: str
    details: Dict[str, Any] = Field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class Setting(BaseModel):
    """Model for settings"""
    setting_id: str
    setting_key: str
    setting_value: Dict[str, Any]
    setting_type: str
    created_at: str
    updated_at: str
    scope: str = "global"  # 'global' or 'user'


class EmbeddingJob(BaseModel):
    """Model for embedding job entries"""
    job_id: str
    status: str  # 'pending', 'processing', 'completed', 'failed'
    document_reference: str
    chunk_count: int
    processed_count: int = 0
    retry_count: int = 0
    failure_reason: Optional[str] = None
    priority: int = 0
    created_at: str
    updated_at: str
    completed_at: Optional[str] = None


class PostgresChunk(BaseModel):
    """Model for chunk data from PostgreSQL"""
    chunk_id: str
    document_reference: str
    page_reference: Optional[int] = None
    section_title: Optional[str] = None
    chunk_text: str
    embedding_id: str
    created_at: datetime
    updated_at: datetime
    processing_version: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_db_row(cls, row: Dict[str, Any]) -> 'PostgresChunk':
        """Create a PostgresChunk from a database row"""
        return cls(
            chunk_id=str(row['chunk_id']),
            document_reference=row['document_reference'],
            page_reference=row.get('page_reference'),
            section_title=row.get('section_title'),
            chunk_text=row['chunk_text'],
            embedding_id=str(row['embedding_id']),
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            processing_version=row['processing_version'],
            metadata=row.get('metadata', {})
        )


class PostgresChatHistory(BaseModel):
    """Model for chat history data from PostgreSQL"""
    chat_id: str
    user_id: Optional[str] = None
    query: str
    response: str
    source_chunks: Optional[List[Dict[str, Any]]] = None
    timestamp: datetime

    @classmethod
    def from_db_row(cls, row: Dict[str, Any]) -> 'PostgresChatHistory':
        """Create a PostgresChatHistory from a database row"""
        return cls(
            chat_id=str(row['chat_id']),
            user_id=str(row['user_id']) if row['user_id'] else None,
            query=row['query'],
            response=row['response'],
            source_chunks=row.get('source_chunks', []),
            timestamp=row['timestamp']
        )


class PostgresLog(BaseModel):
    """Model for log data from PostgreSQL"""
    log_id: str
    user_query: str
    retrieved_chunks: List[Dict[str, Any]]
    response: str
    timestamp: datetime
    retrieval_mode: str

    @classmethod
    def from_db_row(cls, row: Dict[str, Any]) -> 'PostgresLog':
        """Create a PostgresLog from a database row"""
        return cls(
            log_id=str(row['log_id']),
            user_query=row['user_query'],
            retrieved_chunks=row['retrieved_chunks'],
            response=row['response'],
            timestamp=row['timestamp'],
            retrieval_mode=row['retrieval_mode']
        )