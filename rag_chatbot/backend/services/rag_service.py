"""
Service for handling RAG (Retrieval-Augmented Generation) operations.
This service orchestrates the flow between retrieval and generation components.
"""
from typing import List, Dict, Any, Optional
from ..utils.logger import rag_logger
from ..schemas.retrieval import Source
from .retrieval_service import RetrievalService


class RAGService:
    """
    Service for handling RAG (Retrieval-Augmented Generation) operations.
    This service orchestrates the flow between retrieval and generation components.
    """

    def __init__(self):
        self.retrieval_service = RetrievalService()
        # We'll use a placeholder for the LLM service that will be replaced by Agents SDK later
        self.llm_api_key = None
        self.llm_base_url = "https://openrouter.ai/api/v1"
        self.llm_model = "openai/gpt-4-turbo"

    async def initialize(self):
        """
        Initialize the RAG service components.
        """
        await self.retrieval_service.initialize()

    async def generate_response(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a response using the RAG approach.

        Args:
            query: User's query
            top_k: Number of top results to retrieve
            filters: Optional filters for retrieval
            session_id: Optional session ID for conversation context

        Returns:
            Dictionary containing the answer and sources
        """
        try:
            # Retrieve relevant context
            sources = await self.retrieval_service.retrieve_by_query(
                query=query,
                top_k=top_k,
                filters=filters
            )

            if not sources:
                rag_logger.warning(f"No sources found for query: {query[:50]}...")
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "query": query
                }

            # Prepare context from sources
            context_text = "\n\n".join([source.text for source in sources])

            # Generate response using a simple approach (will be replaced by Agents SDK)
            answer = await self._generate_answer_with_context(query, context_text)

            rag_logger.info(f"Generated RAG response for query: {query[:50]}...")
            return {
                "answer": answer,
                "sources": sources,
                "query": query
            }

        except Exception as e:
            rag_logger.error(f"Error in RAG service: {str(e)}")
            return {
                "answer": "An error occurred while processing your request.",
                "sources": [],
                "query": query,
                "error": str(e)
            }

    async def _generate_answer_with_context(self, query: str, context: str) -> str:
        """
        Generate an answer using the provided context.
        This uses the IntelligenceService with Agents SDK.

        Args:
            query: User's query
            context: Retrieved context to use for answering

        Returns:
            Generated answer string
        """
        try:
            # Import the IntelligenceService
            from agents_sdk.services.intelligence_service import IntelligenceService, Source
            from backend.utils.logger import rag_logger

            # Create source objects from the context
            # Since we're getting context as a string, we'll create a single source
            sources = [Source(
                chunk_id="context_chunk_1",
                document_id="retrieved_context",
                text=context[:5000],  # Limit text length
                score=0.9,  # High score since this is our main context
                metadata={"source": "retrieved_context", "length": len(context)}
            )]

            # Initialize and use the IntelligenceService
            intelligence_service = IntelligenceService()
            await intelligence_service.initialize()

            # Process the query with the retrieved context
            result = await intelligence_service.process_query(
                user_query=query,
                context_chunks=sources
            )

            # Extract and return the answer text
            answer = result.get("text", "I couldn't find any relevant information to answer your question.")

            rag_logger.info(f"Generated response using IntelligenceService for query: {query[:50]}...")
            return answer

        except ImportError:
            from backend.utils.logger import rag_logger
            rag_logger.warning("IntelligenceService not available, using basic response generation")
            # Fallback to basic implementation if IntelligenceService is not available
            if len(context.strip()) == 0:
                return "I couldn't find any relevant information to answer your question."

            # Simple context-aware response
            return f"Based on the provided context: {context[:200]}... I can help answer questions about this topic. Please ask your specific question about this content."
        except Exception as e:
            from backend.utils.logger import rag_logger
            rag_logger.error(f"Error in IntelligenceService: {str(e)}")
            # Fallback response
            return "I encountered an error while processing your query. Please try again."

    async def validate_query_and_context(
        self,
        query: str,
        sources: List[Source]
    ) -> bool:
        """
        Validate if the query and context are appropriate for response generation.

        Args:
            query: User's query
            sources: List of retrieved sources

        Returns:
            bool: True if query and context are valid
        """
        if not query or len(query.strip()) == 0:
            rag_logger.warning("Query validation failed: empty query")
            return False

        if len(sources) == 0:
            rag_logger.warning("Context validation failed: no sources provided")
            return False

        return True