"""
Re-embedding functionality for detecting changes and selectively updating embeddings
"""
import asyncio
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from .base_classes import Chunk
from .database import DatabaseManager
from .config import validate_config

logger = logging.getLogger(__name__)


@dataclass
class ChangeDetectionResult:
    """Result of change detection process"""
    new_chunks: List[Chunk]
    modified_chunks: List[Chunk]
    unchanged_chunks: List[Chunk]
    deleted_chunk_ids: List[str]


class ChangeDetector:
    """Class for detecting changes in documents to enable selective re-embedding"""

    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager

    async def detect_changes(
        self,
        new_chunks: List[Chunk],
        document_reference: str
    ) -> ChangeDetectionResult:
        """
        Detect changes between new chunks and existing chunks in the database

        Args:
            new_chunks: New chunks to compare
            document_reference: Reference to the document being processed

        Returns:
            ChangeDetectionResult with categorized chunks
        """
        # Get existing chunks from database for this document
        existing_chunks = await self._get_existing_chunks_for_document(document_reference)

        new_chunks_map = {chunk.chunk_id: chunk for chunk in new_chunks}
        existing_chunks_map = {chunk.chunk_id: chunk for chunk in existing_chunks}

        new_list = []
        modified_list = []
        unchanged_list = []

        # Check each new chunk against existing
        for chunk_id, new_chunk in new_chunks_map.items():
            if chunk_id not in existing_chunks_map:
                # New chunk
                new_list.append(new_chunk)
            else:
                # Check if content has changed by comparing hashes
                existing_chunk = existing_chunks_map[chunk_id]
                if self._content_changed(new_chunk, existing_chunk):
                    # Content has changed
                    modified_list.append(new_chunk)
                else:
                    # Content unchanged
                    unchanged_list.append(new_chunk)

        # Find deleted chunks (existing chunks not in new chunks)
        deleted_ids = [
            chunk_id for chunk_id in existing_chunks_map
            if chunk_id not in new_chunks_map
        ]

        logger.info(
            f"Change detection completed: {len(new_list)} new, "
            f"{len(modified_list)} modified, {len(unchanged_list)} unchanged, "
            f"{len(deleted_ids)} deleted"
        )

        return ChangeDetectionResult(
            new_chunks=new_list,
            modified_chunks=modified_list,
            unchanged_chunks=unchanged_list,
            deleted_chunk_ids=deleted_ids
        )

    def _content_changed(self, chunk1: Chunk, chunk2: Chunk) -> bool:
        """Check if chunk content has changed by comparing hashes"""
        return chunk1.content_hash != chunk2.content_hash

    async def _get_existing_chunks_for_document(self, document_reference: str) -> List[Chunk]:
        """Get existing chunks from database for a specific document"""
        # This would typically query the database to get existing chunks
        # For now, we'll return an empty list as a placeholder
        # In a full implementation, this would fetch from the Neon database
        metadata = await self.database_manager.neon_db.get_metadata_by_chunk_id(document_reference)
        # This is a simplified implementation - in practice, we'd need a method
        # to retrieve all chunks for a document
        return []


class SelectiveReembedder:
    """Class for selectively re-embedding only changed content"""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.change_detector = ChangeDetector(pipeline.database_manager)

    async def selective_reembed(
        self,
        content: str,
        document_reference: str
    ) -> Dict[str, Any]:
        """
        Selectively re-embed content by detecting changes

        Args:
            content: New content to process
            document_reference: Reference to the document

        Returns:
            Dictionary with processing results
        """
        try:
            # Preprocess content
            processed_content, validation_errors = self.pipeline.text_preprocessor.preprocess(content)

            if validation_errors:
                logger.warning(f"Text preprocessing validation errors: {validation_errors}")

            # Chunk the content
            new_chunks = self.pipeline.chunk_processor.chunk_text(processed_content, document_reference)

            logger.info(f"Created {len(new_chunks)} chunks from content")

            # Detect changes
            change_result = await self.change_detector.detect_changes(new_chunks, document_reference)

            # Process only changed and new chunks
            all_embeddings = []
            total_processed = 0

            # Process new chunks
            if change_result.new_chunks:
                logger.info(f"Processing {len(change_result.new_chunks)} new chunks")
                new_embeddings = await self.pipeline.embedding_processor.generate_embeddings(
                    change_result.new_chunks
                )
                all_embeddings.extend(new_embeddings)
                total_processed += len(change_result.new_chunks)

            # Process modified chunks
            if change_result.modified_chunks:
                logger.info(f"Processing {len(change_result.modified_chunks)} modified chunks")
                modified_embeddings = await self.pipeline.embedding_processor.generate_embeddings(
                    change_result.modified_chunks
                )
                all_embeddings.extend(modified_embeddings)
                total_processed += len(change_result.modified_chunks)

            # Delete embeddings for removed chunks
            if change_result.deleted_chunk_ids:
                logger.info(f"Deleting {len(change_result.deleted_chunk_ids)} removed chunks")
                await self._delete_embeddings(change_result.deleted_chunk_ids)

            # Store only the new/modified embeddings
            if all_embeddings:
                store_success = await self.pipeline.database_manager.store_embeddings_with_metadata(
                    change_result.new_chunks + change_result.modified_chunks,
                    all_embeddings
                )

                if not store_success:
                    return {
                        'success': False,
                        'error': "Failed to store updated embeddings"
                    }

            return {
                'success': True,
                'chunks_processed': total_processed,
                'new_chunks': len(change_result.new_chunks),
                'modified_chunks': len(change_result.modified_chunks),
                'unchanged_chunks': len(change_result.unchanged_chunks),
                'deleted_chunks': len(change_result.deleted_chunk_ids),
                'embeddings_generated': len(all_embeddings),
                'validation_errors': validation_errors
            }

        except Exception as e:
            logger.error(f"Error in selective re-embedding: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    async def _delete_embeddings(self, chunk_ids: List[str]) -> bool:
        """Delete embeddings for specified chunk IDs"""
        # This would typically delete from both Qdrant and Neon databases
        # For now, this is a placeholder implementation
        logger.info(f"Deleting embeddings for {len(chunk_ids)} chunks: {chunk_ids}")
        # In a full implementation, we would delete from both databases
        return True