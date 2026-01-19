import asyncio
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from .qdrant.qdrant_client import QdrantClientWrapper
from .qdrant.qdrant_collection_manager import QdrantCollectionManager
from .qdrant.qdrant_utils import QdrantUtils
from .postgres.pg_client import PostgresClient
from .postgres.pg_queries import PostgresQueries
from .postgres.pg_utils import PostgresUtils

from rag_core.interfaces.database_interface import DatabaseInterface, ChunkMetadata, LogEntry, ChatHistoryEntry
from rag_core.utils.logger import rag_logger
from rag_core.utils.timing import timing_decorator


class DatabaseManager(DatabaseInterface):
    """
    Main database manager that provides unified access to both Qdrant and PostgreSQL.
    Implements the DatabaseInterface contract for the RAG system.
    """

    def __init__(self):
        self.qdrant_client = QdrantClientWrapper()
        self.qdrant_manager = QdrantCollectionManager(self.qdrant_client)
        self.qdrant_utils = QdrantUtils()

        self.postgres_client = PostgresClient()
        self.postgres_queries = PostgresQueries()
        self.postgres_utils = PostgresUtils()

        # Flag to track if components are initialized
        self._initialized = False

    async def connect_all(self):
        """
        Initialize connections to both Qdrant and PostgreSQL.
        """
        try:
            # Connect to Qdrant
            await self.qdrant_client.connect()
            await self.qdrant_manager.qdrant_client.ensure_collection_exists()

            # Connect to PostgreSQL
            await self.postgres_client.connect()

            # Verify both connections are working
            qdrant_ok = True  # Qdrant doesn't have a simple health check method
            postgres_ok = await self.postgres_client.health_check()

            if qdrant_ok and postgres_ok:
                self._initialized = True
                rag_logger.info("DatabaseManager: Both Qdrant and PostgreSQL connections established successfully")
            else:
                rag_logger.error("DatabaseManager: Failed to establish all connections")
                raise Exception("Failed to establish all database connections")

        except Exception as e:
            rag_logger.error(f"DatabaseManager: Error connecting to databases: {str(e)}")
            raise

    async def close_all(self):
        """
        Close connections to both Qdrant and PostgreSQL.
        """
        try:
            await self.qdrant_client.disconnect()
            await self.postgres_client.disconnect()
            self._initialized = False
            rag_logger.info("DatabaseManager: All database connections closed")
        except Exception as e:
            rag_logger.error(f"DatabaseManager: Error closing database connections: {str(e)}")
            raise

    def qdrant(self) -> QdrantClientWrapper:
        """
        Get access to Qdrant client.
        """
        if not self._initialized:
            raise RuntimeError("DatabaseManager not initialized. Call connect_all() first.")
        return self.qdrant_client

    def postgres(self) -> PostgresClient:
        """
        Get access to PostgreSQL client.
        """
        if not self._initialized:
            raise RuntimeError("DatabaseManager not initialized. Call connect_all() first.")
        return self.postgres_client

    @timing_decorator
    async def store_chunk_metadata(self, chunk_metadata: ChunkMetadata) -> bool:
        """
        Store chunk metadata in PostgreSQL.
        """
        try:
            if not self._initialized:
                raise RuntimeError("DatabaseManager not initialized. Call connect_all() first.")

            # Prepare parameters for PostgreSQL query
            params = self.postgres_queries.get_insert_chunk_params(
                chunk_id=chunk_metadata.chunk_id,
                document_reference=chunk_metadata.document_reference,
                chunk_text=chunk_metadata.chunk_text,
                embedding_id=chunk_metadata.embedding_id,
                processing_version=chunk_metadata.processing_version,
                page_reference=chunk_metadata.page_reference,
                section_title=chunk_metadata.section_title,
                metadata=chunk_metadata.metadata
            )

            # Execute the query
            result = await self.postgres_client.execute_command(
                self.postgres_queries.INSERT_CHUNK,
                *params
            )

            rag_logger.info(f"Stored chunk metadata for chunk_id: {chunk_metadata.chunk_id}")
            return True

        except Exception as e:
            rag_logger.error(f"Error storing chunk metadata: {str(e)}")
            return False

    @timing_decorator
    async def store_batch_chunks(self, chunk_metadatas: List[ChunkMetadata]) -> bool:
        """
        Store multiple chunk metadata entries in PostgreSQL.
        """
        try:
            if not self._initialized:
                raise RuntimeError("DatabaseManager not initialized. Call connect_all() first.")

            # Prepare batch parameters
            args_list = []
            for chunk_meta in chunk_metadatas:
                params = self.postgres_queries.get_insert_chunk_params(
                    chunk_id=chunk_meta.chunk_id,
                    document_reference=chunk_meta.document_reference,
                    chunk_text=chunk_meta.chunk_text,
                    embedding_id=chunk_meta.embedding_id,
                    processing_version=chunk_meta.processing_version,
                    page_reference=chunk_meta.page_reference,
                    section_title=chunk_meta.section_title,
                    metadata=chunk_meta.metadata
                )
                args_list.append(params)

            # Execute batch insert
            result = await self.postgres_client.execute_many(
                self.postgres_queries.INSERT_CHUNK_MANY,
                args_list
            )

            rag_logger.info(f"Stored batch of {len(chunk_metadatas)} chunk metadata entries")
            return True

        except Exception as e:
            rag_logger.error(f"Error storing batch chunks: {str(e)}")
            return False

    @timing_decorator
    async def get_chunk_metadata(self, chunk_id: str) -> Optional[ChunkMetadata]:
        """
        Retrieve chunk metadata by ID from PostgreSQL.
        """
        try:
            if not self._initialized:
                raise RuntimeError("DatabaseManager not initialized. Call connect_all() first.")

            results = await self.postgres_client.execute_query(
                self.postgres_queries.SELECT_CHUNK_BY_ID,
                chunk_id
            )

            if results:
                row = results[0]
                # Convert the database row to our ChunkMetadata model
                chunk = ChunkMetadata(
                    chunk_id=str(row['chunk_id']),
                    document_reference=row['document_reference'],
                    page_reference=row.get('page_reference'),
                    section_title=row.get('section_title'),
                    chunk_text=row['chunk_text'],
                    embedding_id=str(row['embedding_id']),
                    processing_version=row['processing_version'],
                    created_at=row['created_at'].isoformat() if hasattr(row['created_at'], 'isoformat') else str(row['created_at']),
                    updated_at=row['updated_at'].isoformat() if hasattr(row['updated_at'], 'isoformat') else str(row['updated_at']),
                    metadata=row.get('metadata', {})
                )
                return chunk

            return None

        except Exception as e:
            rag_logger.error(f"Error getting chunk metadata by ID {chunk_id}: {str(e)}")
            return None

    @timing_decorator
    async def get_chunks_by_document(self, document_reference: str) -> List[ChunkMetadata]:
        """
        Retrieve all chunks for a specific document from PostgreSQL.
        """
        try:
            if not self._initialized:
                raise RuntimeError("DatabaseManager not initialized. Call connect_all() first.")

            results = await self.postgres_client.execute_query(
                self.postgres_queries.SELECT_CHUNKS_BY_DOCUMENT,
                document_reference
            )

            chunks = []
            for row in results:
                chunk = ChunkMetadata(
                    chunk_id=str(row['chunk_id']),
                    document_reference=row['document_reference'],
                    page_reference=row.get('page_reference'),
                    section_title=row.get('section_title'),
                    chunk_text=row['chunk_text'],
                    embedding_id=str(row['embedding_id']),
                    processing_version=row['processing_version'],
                    created_at=row['created_at'].isoformat() if hasattr(row['created_at'], 'isoformat') else str(row['created_at']),
                    updated_at=row['updated_at'].isoformat() if hasattr(row['updated_at'], 'isoformat') else str(row['updated_at']),
                    metadata=row.get('metadata', {})
                )
                chunks.append(chunk)

            rag_logger.info(f"Retrieved {len(chunks)} chunks for document: {document_reference}")
            return chunks

        except Exception as e:
            rag_logger.error(f"Error getting chunks by document {document_reference}: {str(e)}")
            return []

    @timing_decorator
    async def log_query(self, log_entry: LogEntry) -> bool:
        """
        Log a query and its results to PostgreSQL.
        """
        try:
            if not self._initialized:
                raise RuntimeError("DatabaseManager not initialized. Call connect_all() first.")

            params = self.postgres_queries.get_insert_log_params(
                log_id=log_entry.log_id,
                user_query=log_entry.user_query,
                retrieved_chunks=log_entry.retrieved_chunks,
                response=log_entry.response,
                retrieval_mode=log_entry.retrieval_mode
            )

            result = await self.postgres_client.execute_command(
                self.postgres_queries.INSERT_LOG,
                *params
            )

            rag_logger.info(f"Logged query with log_id: {log_entry.log_id}")
            return True

        except Exception as e:
            rag_logger.error(f"Error logging query: {str(e)}")
            return False

    @timing_decorator
    async def store_chat_history(self, chat_entry: ChatHistoryEntry) -> bool:
        """
        Store a chat history entry in PostgreSQL.
        """
        try:
            if not self._initialized:
                raise RuntimeError("DatabaseManager not initialized. Call connect_all() first.")

            params = self.postgres_queries.get_insert_chat_message_params(
                chat_id=chat_entry.chat_id,
                query=chat_entry.query,
                response=chat_entry.response,
                user_id=chat_entry.user_id,
                source_chunks=chat_entry.source_chunks
            )

            result = await self.postgres_client.execute_command(
                self.postgres_queries.INSERT_CHAT_MESSAGE,
                *params
            )

            rag_logger.info(f"Stored chat history entry with chat_id: {chat_entry.chat_id}")
            return True

        except Exception as e:
            rag_logger.error(f"Error storing chat history: {str(e)}")
            return False

    @timing_decorator
    async def get_chat_history(
        self,
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[ChatHistoryEntry]:
        """
        Retrieve chat history from PostgreSQL.
        """
        try:
            if not self._initialized:
                raise RuntimeError("DatabaseManager not initialized. Call connect_all() first.")

            if user_id:
                # Get chat history for a specific user
                results = await self.postgres_client.execute_query(
                    self.postgres_queries.SELECT_CHAT_HISTORY_BY_USER,
                    user_id, limit, 0  # offset 0 for now
                )
            else:
                # Get all chat history (with limit)
                results = await self.postgres_client.execute_query(
                    self.postgres_queries.SELECT_CHAT_HISTORY_ALL,
                    limit, 0  # offset 0 for now
                )

            chat_entries = []
            for row in results:
                chat_entry = ChatHistoryEntry(
                    chat_id=str(row['chat_id']),
                    user_id=str(row['user_id']) if row['user_id'] else None,
                    query=row['query'],
                    response=row['response'],
                    source_chunks=row.get('source_chunks', []),
                    timestamp=row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp'])
                )
                chat_entries.append(chat_entry)

            rag_logger.info(f"Retrieved {len(chat_entries)} chat history entries")
            return chat_entries

        except Exception as e:
            rag_logger.error(f"Error getting chat history: {str(e)}")
            return []

    @timing_decorator
    async def store_embedding(
        self,
        chunk_id: str,
        vector: List[float],
        text: str,
        document_reference: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store an embedding in Qdrant with corresponding metadata in PostgreSQL.
        """
        try:
            if not self._initialized:
                raise RuntimeError("DatabaseManager not initialized. Call connect_all() first.")

            # Validate the embedding vector
            if not self.qdrant_utils.validate_embedding_vector(vector):
                rag_logger.error(f"Invalid embedding vector for chunk {chunk_id}")
                return False

            # Create Qdrant payload
            qdrant_payload = self.qdrant_utils.build_payload_from_chunk(
                text=text,
                document_reference=document_reference,
                chunk_id=chunk_id,
                metadata=metadata
            )

            # Create Qdrant point
            qdrant_point = self.qdrant_utils.build_qdrant_point(
                vector=vector,
                chunk_id=chunk_id,
                payload=qdrant_payload
            )

            # Store in Qdrant
            qdrant_success = await self.qdrant_manager.upsert_points([qdrant_point])

            if not qdrant_success:
                rag_logger.error(f"Failed to store embedding in Qdrant for chunk {chunk_id}")
                return False

            rag_logger.info(f"Stored embedding in Qdrant for chunk {chunk_id}")
            return True

        except Exception as e:
            rag_logger.error(f"Error storing embedding: {str(e)}")
            return False

    @timing_decorator
    async def query_embeddings(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query embeddings from Qdrant and enrich with metadata from PostgreSQL.
        """
        try:
            if not self._initialized:
                raise RuntimeError("DatabaseManager not initialized. Call connect_all() first.")

            # Query Qdrant for similar embeddings
            qdrant_results = await self.qdrant_manager.query_by_vector_similarity(
                query_vector=query_vector,
                top_k=top_k,
                filters=filters
            )

            # Convert RetrievalResult objects to the dictionary format expected by the retrieval service
            results = []
            for result in qdrant_results:  # qdrant_results are now RetrievalResult objects
                result_dict = {
                    'id': result.chunk_id,
                    'payload': {
                        'content': result.text,  # Changed from 'text' to 'content' to match the actual stored field
                        'document_reference': result.document_reference,
                        'page_reference': result.page_reference,
                        'section_title': result.section_title,
                        'metadata': result.metadata or {}
                    },
                    'score': result.score
                }
                results.append(result_dict)

            rag_logger.info(f"Retrieved {len(results)} embedding results from Qdrant")
            return results

        except Exception as e:
            rag_logger.error(f"Error querying embeddings: {str(e)}")
            return []

    @timing_decorator
    async def get_user_history(self, user_id: str, limit: int = 10) -> List[ChatHistoryEntry]:
        """
        Get chat history for a specific user.
        """
        return await self.get_chat_history(user_id=user_id, limit=limit)

    @timing_decorator
    async def store_chat_message(
        self,
        chat_id: str,
        user_id: str,
        query: str,
        response: str,
        source_chunks: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Store a chat message in the database.
        """
        chat_entry = ChatHistoryEntry(
            chat_id=chat_id,
            user_id=user_id,
            query=query,
            response=response,
            source_chunks=source_chunks or [],
            timestamp="TODO: add current timestamp"
        )
        return await self.store_chat_history(chat_entry)


# Global instance for dependency injection
database_manager = DatabaseManager()


@asynccontextmanager
async def get_database_manager():
    """
    Async context manager for getting database manager instance.
    """
    try:
        await database_manager.connect_all()
        yield database_manager
    finally:
        await database_manager.close_all()


async def initialize_database_manager():
    """
    Initialize the database manager with connections to both databases.
    """
    await database_manager.connect_all()
    return database_manager


async def shutdown_database_manager():
    """
    Shutdown the database manager and close all connections.
    """
    await database_manager.close_all()