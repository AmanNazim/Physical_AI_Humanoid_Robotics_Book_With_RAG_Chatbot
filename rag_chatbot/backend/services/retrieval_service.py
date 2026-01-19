"""
Service for handling retrieval operations.
This service integrates with the existing database subsystem to retrieve stored embeddings.
"""
from typing import List, Dict, Any, Optional
import sys
import os
# Add the project root to the Python path to allow absolute imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from databases.database_manager import database_manager
from embedding_pipeline.gemini_client import EmbeddingProcessor, GeminiClient
from ..utils.logger import rag_logger
from ..schemas.retrieval import Source


class RetrievalService:
    """
    Service for handling retrieval operations.
    This service integrates with the existing database subsystem to retrieve stored embeddings.
    """

    def __init__(self):
        self.database_manager = database_manager
        self.embedding_processor = EmbeddingProcessor()

    async def initialize(self):
        """
        Initialize the embedding processor.
        """
        await self.embedding_processor.initialize()

    async def retrieve_by_query(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Source]:
        """
        Retrieve similar content based on a query.

        Args:
            query: Query string to find similar content for
            top_k: Number of top results to retrieve
            filters: Optional filters for the search

        Returns:
            List of Source objects containing the retrieval results
        """
        try:
            # Initialize database if needed
            if not self.database_manager._initialized:
                await self.database_manager.connect_all()

            # Create a temporary chunk object for the query
            class TempChunk:
                def __init__(self, content: str):
                    self.content = content

            # Generate embedding for the query using the embedding processor
            query_chunks = [TempChunk(query)]
            query_embeddings = await self.embedding_processor.generate_embeddings(query_chunks)

            if not query_embeddings or len(query_embeddings) == 0:
                rag_logger.error(f"Failed to generate embedding for query: {query}")
                return []

            query_embedding = query_embeddings[0]  # Get the first (and only) embedding

            # Search in Qdrant for similar vectors
            results = await self.database_manager.query_embeddings(
                query_vector=query_embedding,
                top_k=top_k,
                filters=filters
            )

            # Convert results to Source objects
            sources = []
            for result in results:
                source = Source(
                    chunk_id=result.get('id', ''),
                    document_id=result.get('payload', {}).get('document_id', ''),
                    text=result.get('payload', {}).get('content', '')[:500],  # Changed from 'text' to 'content' - Limit text length
                    score=result.get('score', 0.0),
                    metadata=result.get('payload', {}).get('metadata', {})
                )
                sources.append(source)

            rag_logger.info(f"Retrieved {len(sources)} results for query: {query[:50]}...")
            return sources

        except Exception as e:
            rag_logger.error(f"Error in retrieval service: {str(e)}")
            return []

    async def retrieve_by_document_id(self, document_id: str) -> List[Source]:
        """
        Retrieve all chunks for a specific document.

        Args:
            document_id: ID of the document to retrieve chunks for

        Returns:
            List of Source objects containing the document chunks
        """
        try:
            # Initialize database if needed
            if not self.database_manager._initialized:
                await self.database_manager.connect_all()

            # Get chunks by document from PostgreSQL
            chunks = await self.database_manager.get_chunks_by_document(document_id)

            # Convert chunks to Source objects
            sources = []
            for chunk in chunks:
                source = Source(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_reference,
                    text=chunk.chunk_text[:500],  # Limit text length
                    score=1.0,  # Default score for all chunks of the document
                    metadata=chunk.metadata
                )
                sources.append(source)

            rag_logger.info(f"Retrieved {len(sources)} chunks for document: {document_id}")
            return sources

        except Exception as e:
            rag_logger.error(f"Error retrieving document chunks: {str(e)}")
            return []

    async def validate_query(self, query: str) -> bool:
        """
        Validate query before processing.

        Args:
            query: Query string to validate

        Returns:
            bool: True if query is valid
        """
        if not query or len(query.strip()) == 0:
            rag_logger.warning("Query validation failed: empty query")
            return False

        if len(query.strip()) < 3:
            rag_logger.warning("Query validation failed: query too short")
            return False

        return True