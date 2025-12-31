from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class RetrievalResult(BaseModel):
    """Model for retrieval results"""
    chunk_id: str
    text_content: str
    document_reference: str
    score: float
    metadata: Dict[str, Any]


class RetrieverInterface(ABC):
    """
    Abstract interface for the retriever component of the RAG system.
    Defines the contract for retrieving relevant chunks based on a query.
    """

    @abstractmethod
    async def retrieve_chunks(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant text chunks based on the query.

        Args:
            query: The search query
            top_k: Number of top results to return
            filters: Optional filters to apply to the search

        Returns:
            List of retrieval results with text chunks and metadata
        """
        pass

    @abstractmethod
    async def retrieve_by_document(
        self,
        document_reference: str,
        query: str,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks from a specific document.

        Args:
            document_reference: Reference to the specific document
            query: The search query
            top_k: Number of top results to return

        Returns:
            List of retrieval results
        """
        pass

    @abstractmethod
    async def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 5
    ) -> List[List[RetrievalResult]]:
        """
        Perform batch retrieval for multiple queries.

        Args:
            queries: List of search queries
            top_k: Number of top results to return for each query

        Returns:
            List of lists of retrieval results
        """
        pass