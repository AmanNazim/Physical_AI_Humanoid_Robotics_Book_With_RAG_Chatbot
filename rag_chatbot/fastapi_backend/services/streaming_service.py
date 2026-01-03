from typing import AsyncGenerator, Dict, Any
from datetime import datetime
from fastapi import HTTPException
from ..utils.logging import get_logger
from ..models.response_models import Source

logger = get_logger(__name__)


class StreamingService:
    """
    Service for handling streaming responses.
    Provides utilities for streaming tokens and managing streaming sessions.
    """

    def __init__(self):
        pass

    async def stream_response(
        self,
        query: str,
        context_chunks: list,
        rag_service
    ) -> AsyncGenerator[str, None]:
        """
        Stream response tokens from the RAG service.

        Args:
            query: User's query
            context_chunks: List of context chunks
            rag_service: RAG service instance to generate response

        Yields:
            Streamed response chunks
        """
        try:
            # Stream the response from the RAG service
            async for chunk in rag_service.generate_streaming_response(query, context_chunks):
                # Format the chunk for SSE
                yield f"data: {chunk}\n\n"

        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

    async def stream_chat_response(
        self,
        query: str,
        context_chunks: list,
        rag_service,
        session_id: str = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat response with proper formatting.

        Args:
            query: User's query
            context_chunks: List of context chunks
            rag_service: RAG service instance to generate response
            session_id: Optional session ID

        Yields:
            Formatted chat response chunks
        """
        try:
            # Stream the response from the RAG service
            async for chunk in rag_service.generate_streaming_response(query, context_chunks, session_id):
                # Format the chunk as a JSON object for the frontend
                yield f"data: {chunk}\n\n"

        except Exception as e:
            logger.error(f"Error in streaming chat response: {str(e)}")
            error_data = {
                "type": "error",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            yield f"data: {error_data}\n\n"

    async def stream_sources(
        self,
        sources: list
    ) -> AsyncGenerator[str, None]:
        """
        Stream source information.

        Args:
            sources: List of source chunks

        Yields:
            Source information chunks
        """
        try:
            for source in sources:
                source_data = {
                    "type": "source",
                    "chunk_id": source.chunk_id,
                    "document_id": source.document_id,
                    "text": source.text[:200] + "..." if len(source.text) > 200 else source.text,
                    "score": source.score,
                    "metadata": source.metadata
                }
                yield f"data: {source_data}\n\n"

        except Exception as e:
            logger.error(f"Error in streaming sources: {str(e)}")
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

    async def create_stream_session(self, session_id: str = None) -> str:
        """
        Create a new streaming session.

        Args:
            session_id: Optional session ID to use

        Returns:
            Session ID for the streaming session
        """
        import uuid
        session_id = session_id or str(uuid.uuid4())
        logger.info(f"Created streaming session: {session_id}")
        return session_id

    async def cleanup_stream_session(self, session_id: str):
        """
        Clean up resources for a streaming session.

        Args:
            session_id: Session ID to clean up
        """
        logger.info(f"Cleaning up streaming session: {session_id}")
        # Add any cleanup logic here if needed