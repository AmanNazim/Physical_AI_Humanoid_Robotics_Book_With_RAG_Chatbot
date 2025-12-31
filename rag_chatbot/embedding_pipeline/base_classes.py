"""
Base classes for the Embeddings Pipeline System
Following object-oriented design principles with clear separation of concerns
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import uuid
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    chunk_id: str
    content: str
    token_count: int
    character_start: int
    character_end: int
    token_start: int
    token_end: int
    parent_chunk_id: Optional[str] = None
    overlap_type: str = "none"  # "before", "after", "none"
    document_reference: Optional[str] = None
    page_reference: Optional[int] = None
    section_title: Optional[str] = None
    content_hash: Optional[str] = None


class BaseProcessor(ABC):
    """Base class for all processors in the embeddings pipeline"""

    @abstractmethod
    def process(self, *args, **kwargs):
        """Process the input and return the result"""
        pass


class BaseChunker(BaseProcessor):
    """Base class for chunking operations"""

    @abstractmethod
    def chunk_text(self, text: str, document_reference: Optional[str] = None) -> List[Chunk]:
        """Split text into chunks according to specifications"""
        pass


class BaseEmbeddingGenerator(BaseProcessor):
    """Base class for embedding generation"""

    @abstractmethod
    def generate_embeddings(self, chunks: List[Chunk]) -> List[List[float]]:
        """Generate embeddings for the provided chunks"""
        pass


class BaseDatabaseConnector(BaseProcessor):
    """Base class for database operations"""

    @abstractmethod
    def store_embeddings(self, chunks: List[Chunk], embeddings: List[List[float]]) -> bool:
        """Store embeddings and metadata to the database"""
        pass