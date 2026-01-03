from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
from ..clients import EmbeddingsClient, QdrantClient, PostgresClient
from ..utils.logging import get_logger
from ..models.request_models import IngestTextRequest
from ..models.response_models import IngestionResponse
from ..databases.database_manager import database_manager  # Using existing database manager
from ..embedding_pipeline.chunking_engine import ChunkingEngine  # Using existing chunking engine

logger = get_logger(__name__)


class IngestionService:
    """
    Service for handling document ingestion operations.
    Coordinates between embeddings, Qdrant, and PostgreSQL for document processing.
    """

    def __init__(self):
        self.embeddings_client = EmbeddingsClient()
        self.qdrant_client = QdrantClient()
        self.postgres_client = PostgresClient()
        self.database_manager = database_manager  # Using the existing database manager
        self.chunking_engine = ChunkingEngine()  # Using the existing chunking engine

    async def initialize(self):
        """
        Initialize the ingestion service components.
        """
        await self.embeddings_client.initialize()

    async def ingest_text(self, request: IngestTextRequest) -> IngestionResponse:
        """
        Ingest text content into the system.

        Args:
            request: IngestTextRequest containing the text and metadata

        Returns:
            IngestionResponse with the ingestion results
        """
        start_time = datetime.utcnow()

        try:
            # Generate document ID if not provided
            document_id = request.document_id or str(uuid.uuid4())

            # Chunk the text using the existing chunking engine
            chunks = await self.chunking_engine.chunk_text(request.text)
            logger.info(f"Chunked text into {len(chunks)} chunks for document {document_id}")

            # Extract content from chunks for embedding
            chunk_texts = [chunk.content for chunk in chunks]

            # Generate embeddings for the chunks
            embeddings = await self.embeddings_client.generate_embeddings(
                chunk_texts,
                task_type="RETRIEVAL_DOCUMENT"
            )
            logger.info(f"Generated {len(embeddings)} embeddings for document {document_id}")

            # Prepare payloads for Qdrant
            qdrant_payloads = []
            for i, chunk in enumerate(chunks):
                payload = {
                    "text": chunk.content,
                    "document_id": document_id,
                    "chunk_id": f"{document_id}_chunk_{i}",
                    "source": request.source,
                    "title": request.title,
                    "metadata": request.metadata or {}
                }
                qdrant_payloads.append(payload)

            # Store embeddings in Qdrant
            qdrant_success = await self.qdrant_client.insert_vectors(
                vectors=embeddings,
                payloads=qdrant_payloads
            )

            if not qdrant_success:
                raise Exception("Failed to store vectors in Qdrant")

            # Store document metadata in PostgreSQL
            from ..models.request_models import DocumentMetadata
            doc_metadata = DocumentMetadata(
                document_id=document_id,
                title=request.title,
                source=request.source,
                chunk_count=len(chunks),
                created_at=datetime.utcnow()
            )
            postgres_success = await self.postgres_client.save_document_metadata(doc_metadata)

            if not postgres_success:
                raise Exception("Failed to store document metadata in PostgreSQL")

            # Calculate elapsed time
            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            logger.info(f"Successfully ingested document {document_id} with {len(chunks)} chunks")

            return IngestionResponse(
                status="success",
                document_id=document_id,
                chunks_created=len(chunks),
                vectors_stored=len(embeddings),
                elapsed_ms=elapsed_ms
            )

        except Exception as e:
            logger.error(f"Error ingesting text: {str(e)}")
            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return IngestionResponse(
                status="error",
                document_id=request.document_id or "",
                chunks_created=0,
                vectors_stored=0,
                elapsed_ms=elapsed_ms
            )

    async def ingest_document(self, content: str, title: str, source: str = "manual") -> IngestionResponse:
        """
        Ingest a document with the provided content.

        Args:
            content: Content of the document
            title: Title of the document
            source: Source type of the document

        Returns:
            IngestionResponse with the ingestion results
        """
        request = IngestTextRequest(
            text=content,
            title=title,
            source=source
        )
        return await self.ingest_text(request)

    async def validate_document(self, request: IngestTextRequest) -> bool:
        """
        Validate document before ingestion.

        Args:
            request: IngestTextRequest to validate

        Returns:
            bool: True if document is valid
        """
        if not request.text or len(request.text.strip()) == 0:
            logger.warning("Document validation failed: empty text")
            return False

        if not request.title or len(request.title.strip()) == 0:
            logger.warning("Document validation failed: empty title")
            return False

        # Check if source is valid
        valid_sources = ["manual", "pdf", "md"]
        if request.source not in valid_sources:
            logger.warning(f"Document validation failed: invalid source {request.source}")
            return False

        return True

    async def process_text(self, text: str, title: str, source: str = "manual") -> IngestionResponse:
        """
        Process text for ingestion (validation + ingestion).

        Args:
            text: Text content to process
            title: Title of the document
            source: Source type of the document

        Returns:
            IngestionResponse with the processing results
        """
        request = IngestTextRequest(
            text=text,
            title=title,
            source=source
        )

        # Validate the document
        if not await self.validate_document(request):
            return IngestionResponse(
                status="error",
                document_id="",
                chunks_created=0,
                vectors_stored=0,
                elapsed_ms=0
            )

        # Ingest the document
        return await self.ingest_text(request)