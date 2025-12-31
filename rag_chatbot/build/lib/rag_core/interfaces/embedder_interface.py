from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pydantic import BaseModel


class EmbeddingResult(BaseModel):
    """Model for embedding results"""
    chunk_id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]


class EmbedderInterface(ABC):
    """
    Abstract interface for the embedder component of the RAG system.
    Defines the contract for generating embeddings from text chunks.
    """

    @abstractmethod
    async def embed_text(
        self,
        text: str
    ) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as a list of floats
        """
        pass

    @abstractmethod
    async def embed_texts(
        self,
        texts: List[str]
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding results with text and embeddings
        """
        pass

    @abstractmethod
    async def embed_chunk(
        self,
        text: str,
        chunk_id: str,
        metadata: Dict[str, Any]
    ) -> EmbeddingResult:
        """
        Generate embedding for a text chunk with metadata.

        Args:
            text: Input text to embed
            chunk_id: Unique identifier for the chunk
            metadata: Additional metadata for the chunk

        Returns:
            Embedding result with chunk information
        """
        pass