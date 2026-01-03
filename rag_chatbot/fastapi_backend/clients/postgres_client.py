from typing import List, Dict, Any, Optional
import asyncpg
from ..config import settings
from ..utils.logging import get_logger
from ..models.request_models import DocumentMetadata

logger = get_logger(__name__)


class PostgresClient:
    """
    PostgreSQL client wrapper for the FastAPI backend.
    This client provides a clean interface to interact with the Neon Postgres database.
    """

    def __init__(self):
        self.database_url = settings.neon_postgres_url
        self.pool = None

    async def connect(self):
        """
        Establish connection to the PostgreSQL database.
        """
        try:
            self.pool = await asyncpg.create_pool(
                dsn=self.database_url,
                min_size=1,
                max_size=10,
                command_timeout=60
            )
            logger.info("Connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL: {str(e)}")
            raise

    async def disconnect(self):
        """
        Close connection to the PostgreSQL database.
        """
        if self.pool:
            await self.pool.close()
            logger.info("Disconnected from PostgreSQL database")

    async def health_check(self) -> bool:
        """
        Perform a health check on the PostgreSQL connection.

        Returns:
            bool: True if connection is healthy
        """
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                return result == 1
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {str(e)}")
            return False

    async def save_document_metadata(self, document: DocumentMetadata) -> bool:
        """
        Save document metadata to the PostgreSQL database.

        Args:
            document: Document metadata to save

        Returns:
            bool: True if save was successful
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO documents (document_id, title, source, chunk_count, created_at)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (document_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        source = EXCLUDED.source,
                        chunk_count = EXCLUDED.chunk_count,
                        updated_at = NOW()
                    """,
                    document.document_id,
                    document.title,
                    document.source,
                    document.chunk_count,
                    document.created_at
                )
                logger.info(f"Saved document metadata: {document.document_id}")
                return True
        except Exception as e:
            logger.error(f"Error saving document metadata: {str(e)}")
            return False

    async def get_document_metadata(self, document_id: str) -> Optional[DocumentMetadata]:
        """
        Retrieve document metadata from the PostgreSQL database.

        Args:
            document_id: ID of the document to retrieve

        Returns:
            DocumentMetadata object if found, None otherwise
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT document_id, title, source, chunk_count, created_at
                    FROM documents
                    WHERE document_id = $1
                    """,
                    document_id
                )

                if row:
                    return DocumentMetadata(
                        document_id=row['document_id'],
                        title=row['title'],
                        source=row['source'],
                        chunk_count=row['chunk_count'],
                        created_at=row['created_at']
                    )
                return None
        except Exception as e:
            logger.error(f"Error getting document metadata: {str(e)}")
            return None

    async def get_all_documents(self) -> List[DocumentMetadata]:
        """
        Retrieve all document metadata from the PostgreSQL database.

        Returns:
            List of DocumentMetadata objects
        """
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT document_id, title, source, chunk_count, created_at
                    FROM documents
                    ORDER BY created_at DESC
                    """
                )

                documents = []
                for row in rows:
                    document = DocumentMetadata(
                        document_id=row['document_id'],
                        title=row['title'],
                        source=row['source'],
                        chunk_count=row['chunk_count'],
                        created_at=row['created_at']
                    )
                    documents.append(document)

                logger.info(f"Retrieved {len(documents)} documents from PostgreSQL")
                return documents
        except Exception as e:
            logger.error(f"Error getting all documents: {str(e)}")
            return []

    async def save_chat_history(
        self,
        chat_id: str,
        user_id: str,
        query: str,
        response: str,
        source_chunks: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Save chat history to the PostgreSQL database.

        Args:
            chat_id: Unique identifier for the chat session
            user_id: ID of the user
            query: User's query
            response: AI's response
            source_chunks: Optional list of source chunks used

        Returns:
            bool: True if save was successful
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO chat_history (chat_id, user_id, query, response, source_chunks, timestamp)
                    VALUES ($1, $2, $3, $4, $5, NOW())
                    """,
                    chat_id,
                    user_id,
                    query,
                    response,
                    source_chunks or []
                )
                logger.info(f"Saved chat history: {chat_id}")
                return True
        except Exception as e:
            logger.error(f"Error saving chat history: {str(e)}")
            return False

    async def get_chat_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve chat history for a specific user.

        Args:
            user_id: ID of the user
            limit: Maximum number of history entries to return

        Returns:
            List of chat history entries
        """
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT chat_id, query, response, source_chunks, timestamp
                    FROM chat_history
                    WHERE user_id = $1
                    ORDER BY timestamp DESC
                    LIMIT $2
                    """,
                    user_id,
                    limit
                )

                history = []
                for row in rows:
                    history_entry = {
                        "chat_id": row['chat_id'],
                        "query": row['query'],
                        "response": row['response'],
                        "source_chunks": row['source_chunks'],
                        "timestamp": row['timestamp']
                    }
                    history.append(history_entry)

                logger.info(f"Retrieved {len(history)} chat history entries for user: {user_id}")
                return history
        except Exception as e:
            logger.error(f"Error getting chat history: {str(e)}")
            return []

    async def log_query(
        self,
        user_query: str,
        retrieved_chunks: List[Dict[str, Any]],
        response: str,
        retrieval_mode: str = "vector"
    ) -> bool:
        """
        Log a query and its results to the PostgreSQL database.

        Args:
            user_query: The original user query
            retrieved_chunks: List of chunks retrieved during processing
            response: The final response generated
            retrieval_mode: The mode used for retrieval (vector, keyword, hybrid)

        Returns:
            bool: True if logging was successful
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO query_logs (user_query, retrieved_chunks, response, retrieval_mode, timestamp)
                    VALUES ($1, $2, $3, $4, NOW())
                    """,
                    user_query,
                    retrieved_chunks,
                    response,
                    retrieval_mode
                )
                logger.info("Logged query to PostgreSQL")
                return True
        except Exception as e:
            logger.error(f"Error logging query: {str(e)}")
            return False