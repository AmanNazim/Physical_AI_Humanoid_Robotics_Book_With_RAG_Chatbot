"""
Database integration for storing embeddings to Qdrant and Neon databases
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import hashlib
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct
import asyncpg
from .config import QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY, NEON_DATABASE_URL, QDRANT_COLLECTION_NAME, EMBEDDING_DIM
from .base_classes import Chunk

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingRecord:
    """Represents an embedding record with metadata"""
    id: str
    chunk_id: str
    embedding: List[float]
    content: str
    document_reference: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class QdrantDatabase:
    """Qdrant vector database integration"""

    def __init__(self):
        # Check if QDRANT_HOST contains protocol (for cloud instances)
        if QDRANT_HOST.startswith(('http://', 'https://')):
            # For cloud instances, use url parameter instead of host/port
            self.client = QdrantClient(
                url=QDRANT_HOST,
                api_key=QDRANT_API_KEY,
            )
        else:
            # For local instances, use host/port
            self.client = QdrantClient(
                host=QDRANT_HOST,
                port=QDRANT_PORT,
                api_key=QDRANT_API_KEY,
                prefer_grpc=True
            )
        self.collection_name = QDRANT_COLLECTION_NAME  # Use the configured collection name

    async def initialize(self):
        """Initialize the Qdrant collection if it doesn't exist"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                # Create collection with vector size based on configuration
                # Make sure to use the correct embedding dimension from config
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=EMBEDDING_DIM,  # Use configured embedding dimension (should be 1536 from .env)
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Qdrant collection {self.collection_name} already exists")

        except Exception as e:
            logger.error(f"Error initializing Qdrant collection: {str(e)}")
            raise

    async def store_embeddings(self, records: List[EmbeddingRecord]) -> bool:
        """Store embeddings to Qdrant database"""
        try:
            points = []
            for record in records:
                point = PointStruct(
                    id=record.id,
                    vector=record.embedding,
                    payload={
                        "chunk_id": record.chunk_id,
                        "content": record.content,
                        "document_reference": record.document_reference,
                        "metadata": record.metadata or {}
                    }
                )
                points.append(point)

            # Process in smaller batches to avoid timeout issues
            batch_size = 10  # Smaller batch size to avoid timeouts
            total_points = len(points)

            for i in range(0, total_points, batch_size):
                batch = points[i:i + batch_size]

                # Upsert batch to Qdrant
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )

                logger.info(f"Stored batch {i//batch_size + 1} of {(total_points + batch_size - 1)//batch_size} to Qdrant")

            logger.info(f"Stored {len(points)} embeddings to Qdrant")
            return True

        except Exception as e:
            logger.error(f"Error storing embeddings to Qdrant: {str(e)}")
            return False

    async def search_similar(self, query_embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar embeddings in Qdrant"""
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit
            )

            return [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "content": hit.payload.get("content", ""),
                    "document_reference": hit.payload.get("document_reference"),
                    "metadata": hit.payload.get("metadata", {})
                }
                for hit in results
            ]

        except Exception as e:
            logger.error(f"Error searching Qdrant: {str(e)}")
            return []

    async def get_embedding_by_id(self, embedding_id: str) -> Optional[EmbeddingRecord]:
        """Retrieve a specific embedding by ID"""
        try:
            records = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[embedding_id]
            )

            if records:
                record = records[0]
                return EmbeddingRecord(
                    id=record.id,
                    chunk_id=record.payload.get("chunk_id"),
                    embedding=record.vector,
                    content=record.payload.get("content", ""),
                    document_reference=record.payload.get("document_reference"),
                    metadata=record.payload.get("metadata", {})
                )

            return None

        except Exception as e:
            logger.error(f"Error retrieving embedding from Qdrant: {str(e)}")
            return None


class NeonDatabase:
    """Neon PostgreSQL database integration for metadata storage"""

    def __init__(self):
        self.database_url = NEON_DATABASE_URL
        self.pool = None

    async def initialize(self):
        """Initialize the database connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=1,
                max_size=10,
                command_timeout=60
            )

            # Create table if it doesn't exist
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS embeddings_metadata (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        chunk_id TEXT NOT NULL,
                        document_reference TEXT,
                        content_hash TEXT NOT NULL,
                        content_length INTEGER,
                        token_count INTEGER,
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW()
                    )
                """)

                # Create indexes
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunk_id ON embeddings_metadata(chunk_id);
                    CREATE INDEX IF NOT EXISTS idx_document_reference ON embeddings_metadata(document_reference);
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_content_hash_unique ON embeddings_metadata(content_hash);
                """)

            logger.info("Neon database initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing Neon database: {str(e)}")
            raise

    async def store_metadata(self, chunk: Chunk, content_hash: str) -> bool:
        """Store metadata for an embedding in Neon database"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO embeddings_metadata
                    (chunk_id, document_reference, content_hash, content_length, token_count)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (content_hash) DO UPDATE SET
                        updated_at = NOW(),
                        token_count = EXCLUDED.token_count
                """,
                    chunk.chunk_id,
                    chunk.document_reference,
                    content_hash,
                    len(chunk.content),
                    chunk.token_count
                )

            logger.info(f"Stored metadata for chunk {chunk.chunk_id}")
            return True

        except Exception as e:
            logger.error(f"Error storing metadata to Neon: {str(e)}")
            return False

    async def batch_store_metadata(self, chunks: List[Chunk]) -> bool:
        """Store metadata for multiple chunks in batch"""
        try:
            async with self.pool.acquire() as conn:
                records = []
                for chunk in chunks:
                    content_hash = hashlib.sha256(chunk.content.encode()).hexdigest()
                    records.append((
                        chunk.chunk_id,
                        chunk.document_reference,
                        content_hash,
                        len(chunk.content),
                        chunk.token_count
                    ))

                await conn.executemany("""
                    INSERT INTO embeddings_metadata
                    (chunk_id, document_reference, content_hash, content_length, token_count)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (content_hash) DO UPDATE SET
                        updated_at = NOW(),
                        token_count = EXCLUDED.token_count
                """, records)

            logger.info(f"Stored metadata for {len(chunks)} chunks")
            return True

        except Exception as e:
            logger.error(f"Error storing batch metadata to Neon: {str(e)}")
            return False

    async def get_metadata_by_chunk_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a specific chunk ID"""
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM embeddings_metadata WHERE chunk_id = $1
                """, chunk_id)

                if row:
                    return dict(row)

            return None

        except Exception as e:
            logger.error(f"Error retrieving metadata from Neon: {str(e)}")
            return None


class DatabaseManager:
    """Manager class for handling both Qdrant and Neon database operations"""

    def __init__(self):
        self.qdrant_db = QdrantDatabase()
        self.neon_db = NeonDatabase()

    async def initialize(self):
        """Initialize both databases"""
        await self.qdrant_db.initialize()
        await self.neon_db.initialize()

    async def store_embeddings_with_metadata(self, chunks: List[Chunk], embeddings: List[List[float]]) -> bool:
        """Store embeddings to Qdrant and metadata to Neon in a coordinated way"""
        if len(chunks) != len(embeddings):
            logger.error(f"Chunk and embedding count mismatch: {len(chunks)} vs {len(embeddings)}")
            return False

        # Create embedding records
        records = []
        for chunk, embedding in zip(chunks, embeddings):
            record = EmbeddingRecord(
                id=str(uuid.uuid4()),
                chunk_id=chunk.chunk_id,
                embedding=embedding,
                content=chunk.content,
                document_reference=chunk.document_reference,
                metadata={
                    "token_count": chunk.token_count,
                    "character_start": chunk.character_start,
                    "character_end": chunk.character_end,
                    "overlap_type": chunk.overlap_type
                }
            )
            records.append(record)

        # Store embeddings to Qdrant
        qdrant_success = await self.qdrant_db.store_embeddings(records)

        # Store metadata to Neon
        neon_success = True
        for chunk in chunks:
            content_hash = hashlib.sha256(chunk.content.encode()).hexdigest()
            metadata_success = await self.neon_db.store_metadata(chunk, content_hash)
            if not metadata_success:
                neon_success = False
                logger.error(f"Failed to store metadata for chunk {chunk.chunk_id}")

        return qdrant_success and neon_success

    async def search_similar(self, query_embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar embeddings across both databases"""
        return await self.qdrant_db.search_similar(query_embedding, limit)

    async def get_embedding_by_id(self, embedding_id: str) -> Optional[EmbeddingRecord]:
        """Retrieve embedding by ID from Qdrant"""
        return await self.qdrant_db.get_embedding_by_id(embedding_id)