from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
from ..clients import EmbeddingsClient, QdrantClient, PostgresClient, IntelligenceClient
from ..utils.logging import get_logger
from ..models.request_models import QueryRequest, ChatRequest
from ..models.response_models import QueryResponse, ChatResponse, Source
from ..databases.database_manager import database_manager  # Using existing database manager

logger = get_logger(__name__)


class QueryService:
    """
    Service for handling query operations.
    Orchestrates the RAG pipeline: retrieval + reasoning.
    """

    def __init__(self):
        self.embeddings_client = EmbeddingsClient()
        self.qdrant_client = QdrantClient()
        self.postgres_client = PostgresClient()
        self.intelligence_client = IntelligenceClient()
        self.database_manager = database_manager  # Using the existing database manager

    async def initialize(self):
        """
        Initialize the query service components.
        """
        await self.embeddings_client.initialize()

    async def validate_query(self, query: str) -> bool:
        """
        Validate query before processing.

        Args:
            query: Query string to validate

        Returns:
            bool: True if query is valid
        """
        if not query or len(query.strip()) == 0:
            logger.warning("Query validation failed: empty query")
            return False

        if len(query.strip()) < 3:
            logger.warning("Query validation failed: query too short")
            return False

        return True

    async def retrieve_context(self, query: str, top_k: int = 5) -> List[Source]:
        """
        Retrieve context for the given query.

        Args:
            query: Query string to retrieve context for
            top_k: Number of top results to retrieve

        Returns:
            List of Source objects containing the retrieved context
        """
        try:
            # Generate embedding for the query
            query_embedding = await self.embeddings_client.embed_single_text(
                query,
                task_type="RETRIEVAL_QUERY"
            )

            # Search in Qdrant for similar vectors
            sources = await self.qdrant_client.search(
                query_vector=query_embedding,
                top_k=top_k
            )

            logger.info(f"Retrieved {len(sources)} context chunks for query: {query[:50]}...")
            return sources

        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []

    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """
        Process a query request through the RAG pipeline.

        Args:
            request: QueryRequest containing the query and parameters

        Returns:
            QueryResponse with the answer and sources
        """
        start_time = datetime.utcnow()

        try:
            # Validate the query
            if not await self.validate_query(request.query):
                return QueryResponse(
                    answer="Invalid query provided",
                    sources=[],
                    latency_ms=0
                )

            # Retrieve context from the vector database
            sources = await self.retrieve_context(request.query, request.max_context)

            # Generate response using the intelligence subsystem
            answer = await self.intelligence_client.generate_response(
                query=request.query,
                context_chunks=sources
            )

            # Calculate latency
            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            logger.info(f"Processed query in {latency_ms:.2f}ms")

            return QueryResponse(
                answer=answer,
                sources=sources,
                latency_ms=latency_ms
            )

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return QueryResponse(
                answer="An error occurred while processing your query",
                sources=[],
                latency_ms=latency_ms
            )

    async def process_chat(self, request: ChatRequest) -> ChatResponse:
        """
        Process a chat request through the RAG pipeline.

        Args:
            request: ChatRequest containing the query and session information

        Returns:
            ChatResponse with the answer, sources, and session info
        """
        start_time = datetime.utcnow()

        try:
            # Validate the query
            if not await self.validate_query(request.query):
                return ChatResponse(
                    answer="Invalid query provided",
                    sources=[],
                    latency_ms=0,
                    session_id=request.session_id
                )

            # Retrieve context from the vector database
            sources = await self.retrieve_context(request.query, request.max_context)

            # Generate response using the intelligence subsystem
            answer = await self.intelligence_client.generate_response(
                query=request.query,
                context_chunks=sources,
                session_id=request.session_id
            )

            # Calculate latency
            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Generate session ID if not provided
            session_id = request.session_id or str(uuid.uuid4())

            # Log the chat interaction
            await self.postgres_client.save_chat_history(
                chat_id=str(uuid.uuid4()),
                user_id="anonymous",  # In a real system, this would come from auth
                query=request.query,
                response=answer,
                source_chunks=[source.model_dump() for source in sources]
            )

            logger.info(f"Processed chat query in {latency_ms:.2f}ms")

            return ChatResponse(
                answer=answer,
                sources=sources,
                latency_ms=latency_ms,
                session_id=session_id
            )

        except Exception as e:
            logger.error(f"Error processing chat: {str(e)}")
            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return ChatResponse(
                answer="An error occurred while processing your chat request",
                sources=[],
                latency_ms=latency_ms,
                session_id=request.session_id
            )

    async def format_response(self, answer: str, sources: List[Source]) -> str:
        """
        Format the response with sources.

        Args:
            answer: The generated answer
            sources: List of sources used

        Returns:
            Formatted response string
        """
        if not sources:
            return answer

        formatted_sources = "\n\nSources:\n"
        for i, source in enumerate(sources[:3], 1):  # Limit to first 3 sources
            formatted_sources += f"[{i}] {source.text[:200]}...\n"

        return f"{answer}{formatted_sources}"