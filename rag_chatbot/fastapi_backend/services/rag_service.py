from typing import List, Dict, Any, Optional
from datetime import datetime
from ..clients import IntelligenceClient
from ..utils.logging import get_logger
from ..models.response_models import Source

logger = get_logger(__name__)


class RAGService:
    """
    Service for handling RAG (Retrieval-Augmented Generation) operations.
    This service orchestrates the flow between retrieval and generation components.
    """

    def __init__(self):
        self.intelligence_client = IntelligenceClient()

    async def generate_response(
        self,
        query: str,
        context_chunks: List[Source],
        session_id: Optional[str] = None
    ) -> str:
        """
        Generate a response using the RAG approach.

        Args:
            query: User's query
            context_chunks: List of context chunks to provide to the LLM
            session_id: Optional session ID for conversation context

        Returns:
            Generated response string
        """
        try:
            # Use the intelligence client to generate the response
            response = await self.intelligence_client.generate_response(
                query=query,
                context_chunks=context_chunks,
                session_id=session_id
            )

            logger.info(f"Generated RAG response for query: {query[:50]}...")
            return response

        except Exception as e:
            logger.error(f"Error in RAG service generate_response: {str(e)}")
            raise

    async def generate_streaming_response(
        self,
        query: str,
        context_chunks: List[Source],
        session_id: Optional[str] = None
    ):
        """
        Generate a streaming response using the RAG approach.

        Args:
            query: User's query
            context_chunks: List of context chunks to provide to the LLM
            session_id: Optional session ID for conversation context

        Yields:
            Streamed tokens/fragments of the response
        """
        try:
            # Use the intelligence client to generate the streaming response
            async for chunk in self.intelligence_client.generate_streaming_response(
                query=query,
                context_chunks=context_chunks,
                session_id=session_id
            ):
                yield chunk

        except Exception as e:
            logger.error(f"Error in RAG service generate_streaming_response: {str(e)}")
            raise

    async def combine_query_and_context(
        self,
        query: str,
        context_chunks: List[Source]
    ) -> str:
        """
        Combine the query and context for LLM processing.

        Args:
            query: User's query
            context_chunks: List of context chunks

        Returns:
            Combined string ready for LLM processing
        """
        # Prepare the context from the source chunks
        context_text = "\n\n".join([chunk.text for chunk in context_chunks])

        # Combine query and context
        combined = f"Context: {context_text}\n\nQuestion: {query}\n\nPlease provide a detailed answer based on the context provided. If the context doesn't contain the information needed to answer the question, say so."

        return combined

    async def validate_context_relevance(
        self,
        query: str,
        context_chunks: List[Source]
    ) -> bool:
        """
        Validate if the context is relevant to the query.

        Args:
            query: User's query
            context_chunks: List of context chunks

        Returns:
            bool: True if context is relevant to the query
        """
        # For now, we'll just check if there's any context
        # In a more sophisticated implementation, we could use semantic similarity
        return len(context_chunks) > 0

    async def format_sources(self, sources: List[Source]) -> str:
        """
        Format the sources for display in the response.

        Args:
            sources: List of source chunks

        Returns:
            Formatted string of sources
        """
        if not sources:
            return ""

        formatted_sources = "\n\nSources:\n"
        for i, source in enumerate(sources, 1):
            # Limit the text length for display
            snippet = source.text[:200] + "..." if len(source.text) > 200 else source.text
            formatted_sources += f"[{i}] {snippet}\n"

        return formatted_sources

    async def process_rag_flow(
        self,
        query: str,
        context_chunks: List[Source],
        session_id: Optional[str] = None
    ) -> str:
        """
        Process the complete RAG flow: validate context, generate response, format output.

        Args:
            query: User's query
            context_chunks: List of context chunks
            session_id: Optional session ID for conversation context

        Returns:
            Complete response with answer and formatted sources
        """
        try:
            # Validate context relevance
            if not await self.validate_context_relevance(query, context_chunks):
                logger.warning("Context may not be relevant to the query")

            # Generate the response
            answer = await self.generate_response(query, context_chunks, session_id)

            # Format the sources
            sources_text = await self.format_sources(context_chunks)

            # Combine answer with sources
            final_response = f"{answer}{sources_text}"

            logger.info(f"Completed RAG flow for query: {query[:50]}...")
            return final_response

        except Exception as e:
            logger.error(f"Error in RAG service process_rag_flow: {str(e)}")
            raise