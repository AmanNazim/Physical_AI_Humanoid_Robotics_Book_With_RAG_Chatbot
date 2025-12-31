from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel


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
    metadata: Dict[str, Any] = {}


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


class DatabaseInterface(ABC):
    """
    Abstract interface for the database component of the RAG system.
    Defines the contract for storing and retrieving metadata, logs, and chat history.
    """

    @abstractmethod
    async def store_chunk_metadata(
        self,
        chunk_metadata: ChunkMetadata
    ) -> bool:
        """
        Store chunk metadata in the database.

        Args:
            chunk_metadata: Metadata for the text chunk

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def store_batch_chunks(
        self,
        chunk_metadatas: List[ChunkMetadata]
    ) -> bool:
        """
        Store multiple chunk metadata entries in the database.

        Args:
            chunk_metadatas: List of chunk metadata to store

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def get_chunk_metadata(
        self,
        chunk_id: str
    ) -> Optional[ChunkMetadata]:
        """
        Retrieve chunk metadata by ID.

        Args:
            chunk_id: ID of the chunk to retrieve

        Returns:
            Chunk metadata if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_chunks_by_document(
        self,
        document_reference: str
    ) -> List[ChunkMetadata]:
        """
        Retrieve all chunks for a specific document.

        Args:
            document_reference: Reference to the document

        Returns:
            List of chunk metadata for the document
        """
        pass

    @abstractmethod
    async def log_query(
        self,
        log_entry: LogEntry
    ) -> bool:
        """
        Log a query and its results.

        Args:
            log_entry: Log entry to store

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def store_chat_history(
        self,
        chat_entry: ChatHistoryEntry
    ) -> bool:
        """
        Store a chat history entry.

        Args:
            chat_entry: Chat history entry to store

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def get_chat_history(
        self,
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[ChatHistoryEntry]:
        """
        Retrieve chat history.

        Args:
            user_id: Optional user ID to filter by
            limit: Maximum number of entries to return

        Returns:
            List of chat history entries
        """
        pass