from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class AgentResponse(BaseModel):
    """Model for agent responses"""
    response: str
    citations: List[str]
    reasoning_path: Optional[str] = None
    confidence_score: Optional[float] = None


class AgentContext(BaseModel):
    """Model for agent context"""
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    mode: str  # 'full_book' or 'selected_text_only'
    metadata: Dict[str, Any] = {}


class AgentInterface(ABC):
    """
    Abstract interface for the agent component of the RAG system.
    Defines the contract for reasoning and answer generation based on context.
    """

    @abstractmethod
    async def process_query(
        self,
        context: AgentContext
    ) -> AgentResponse:
        """
        Process a query with the provided context.

        Args:
            context: Context containing query, retrieved chunks, and metadata

        Returns:
            Agent response with answer and citations
        """
        pass

    @abstractmethod
    async def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        mode: str = 'full_book'
    ) -> AgentResponse:
        """
        Generate an answer based on the query and context chunks.

        Args:
            query: User query
            context_chunks: Retrieved context chunks
            mode: Retrieval mode ('full_book' or 'selected_text_only')

        Returns:
            Agent response with answer and citations
        """
        pass

    @abstractmethod
    async def validate_response(
        self,
        response: AgentResponse,
        context: AgentContext
    ) -> bool:
        """
        Validate that the response is grounded in the provided context.

        Args:
            response: Agent response to validate
            context: Original context used for generation

        Returns:
            True if response is valid and grounded, False otherwise
        """
        pass

    @abstractmethod
    async def get_reasoning_path(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Get the reasoning path for how the answer was derived.

        Args:
            query: User query
            context_chunks: Context chunks used for reasoning

        Returns:
            Explanation of the reasoning path
        """
        pass