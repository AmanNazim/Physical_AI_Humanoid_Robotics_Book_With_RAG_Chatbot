"""
Service for handling embedding operations.
This service coordinates with the existing embeddings subsystem.
"""
from typing import List, Dict, Any, Optional
import sys
import os
# Add the project root to the Python path to allow absolute imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from databases.database_manager import database_manager
from embedding_pipeline.pipeline import EmbeddingPipeline
from embedding_pipeline.gemini_client import EmbeddingProcessor
from ..utils.logger import rag_logger
from ..schemas.embedding import EmbeddingRequest, EmbeddingResponse


class EmbeddingService:
    """
    Service for handling embedding operations.
    Integrates with the existing embedding pipeline.
    """

    def __init__(self):
        self.initialized = False
        self._embedding_pipeline = EmbeddingPipeline()
        self._database_manager = database_manager

    async def initialize(self):
        """Initialize the embedding service."""
        if not self.initialized:
            await self._embedding_pipeline.initialize()
            if not self._database_manager._initialized:
                await self._database_manager.connect_all()
            self.initialized = True
            rag_logger.info("EmbeddingService initialized successfully")

    async def validate_document(self, text: str) -> bool:
        """
        Validate document before processing.

        Args:
            text: Document text to validate

        Returns:
            True if document is valid, False otherwise
        """
        if not text or len(text.strip()) == 0:
            return False
        if len(text.strip()) < 10:  # Minimum content length
            return False
        return True

    async def trigger_ingestion(self, text: str, document_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trigger the embedding ingestion process.

        Args:
            text: Text content to embed
            document_metadata: Metadata about the document

        Returns:
            Result of the ingestion process
        """
        if not self.initialized:
            await self.initialize()

        try:
            # Process the content through the existing pipeline
            result = await self._embedding_pipeline.process_content(
                content=text,
                document_reference=document_metadata.get("title", "unknown")
            )

            if result['success']:
                rag_logger.info(f"Successfully ingested document: {document_metadata.get('title', 'unknown')}")
                return {
                    "status": "success",
                    "message": "Document successfully embedded and stored",
                    "chunks_processed": result.get('chunks_processed', 0),
                    "embeddings_generated": result.get('embeddings_generated', 0)
                }
            else:
                rag_logger.error(f"Failed to ingest document: {result.get('error', 'Unknown error')}")
                return {
                    "status": "error",
                    "message": result.get('error', 'Unknown error during ingestion')
                }

        except Exception as e:
            rag_logger.error(f"Error in trigger_ingestion: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }