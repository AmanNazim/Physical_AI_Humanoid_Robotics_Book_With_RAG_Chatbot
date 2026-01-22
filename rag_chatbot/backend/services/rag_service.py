"""
Service for handling RAG (Retrieval-Augmented Generation) operations.
This service orchestrates the flow between retrieval and generation components.
"""
from typing import List, Dict, Any, Optional
import json
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

            # Instead of preparing context as a string, pass the sources directly to IntelligenceService
            # Generate response using IntelligenceService with actual sources
            answer = await self._generate_answer_with_sources(query, sources)

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

    async def _generate_answer_with_sources(self, query: str, sources: List[Source]) -> str:
        """
        Generate an answer using the provided sources.
        This uses the IntelligenceService with Agents SDK.

        Args:
            query: User's query
            sources: Retrieved sources to use for answering

        Returns:
            Generated answer string
        """
        try:
            # Import the IntelligenceService
            from agents_sdk.services.intelligence_service import IntelligenceService
            from backend.utils.logger import rag_logger

            # Log before initialization to help debug
            rag_logger.info(f"Initializing IntelligenceService for query: {query[:50]}...")

            # Initialize and use the IntelligenceService
            intelligence_service = IntelligenceService()

            try:
                await intelligence_service.initialize()
                rag_logger.info(f"IntelligenceService initialized successfully for query: {query[:50]}...")
            except Exception as init_error:
                rag_logger.error(f"IntelligenceService initialization failed: {str(init_error)}")
                rag_logger.error(f"Initialization exception type: {type(init_error).__name__}")
                import traceback
                rag_logger.error(f"Initialization full traceback: {traceback.format_exc()}")
                raise init_error

            # Process the query with the retrieved sources directly
            result = await intelligence_service.process_query(
                user_query=query,
                context_chunks=sources  # Pass actual sources instead of context string
            )

            # Extract and return the answer text
            answer = result.get("text", "I couldn't find any relevant information to answer your question.")

            rag_logger.info(f"Generated response using IntelligenceService for query: {query[:50]}...")
            return answer

        except ImportError as e:
            from backend.utils.logger import rag_logger
            rag_logger.error(f"IntelligenceService import failed with error: {str(e)}")
            rag_logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            rag_logger.error(f"Full traceback: {traceback.format_exc()}")

            # Try to identify the specific missing module
            try:
                import agents
                rag_logger.info("agents module is available")
            except ImportError as agent_import_error:
                rag_logger.error(f"agents module import failed: {str(agent_import_error)}")

            try:
                from agents_sdk.services.intelligence_service import IntelligenceService
                rag_logger.info("IntelligenceService import worked separately")
            except ImportError as intelligence_import_error:
                rag_logger.error(f"Direct IntelligenceService import failed: {str(intelligence_import_error)}")

            rag_logger.warning("IntelligenceService not available, using basic response generation")
            # Fallback to basic implementation if IntelligenceService is not available
            if len(sources) == 0:
                return "I couldn't find any relevant information to answer your question."

            # Simple context-aware response using the first source
            if sources:
                context_preview = sources[0].text[:200] if sources[0].text else ""
                return f"Based on the provided context: {context_preview}... I can help answer questions about this topic. Please ask your specific question about this content."
            else:
                return "I couldn't find any relevant information to answer your question."
        except Exception as e:
            from backend.utils.logger import rag_logger
            rag_logger.error(f"Error in IntelligenceService: {str(e)}")
            rag_logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            rag_logger.error(f"Full traceback: {traceback.format_exc()}")
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

    async def stream_response(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ):
        """
        Stream response using the IntelligenceService's streaming capability.

        Args:
            query: User's query
            top_k: Number of top results to retrieve
            filters: Optional filters for retrieval
            session_id: Optional session ID for conversation context

        Yields:
            Streaming response chunks
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
                # Yield a message indicating no sources found
                yield f"data: {json.dumps({'type': 'token', 'content': 'I couldn\'t find any relevant information to answer your question.'})}\n\n"
                yield f"data: {json.dumps({'type': 'complete', 'message': 'Response completed'})}\n\n"
                return

            # Import the IntelligenceService
            from agents_sdk.services.intelligence_service import IntelligenceService
            from backend.utils.logger import rag_logger
            import json

            # Log before initialization to help debug
            rag_logger.info(f"Initializing IntelligenceService for streaming query: {query[:50]}...")

            # Initialize and use the IntelligenceService
            intelligence_service = IntelligenceService()

            try:
                await intelligence_service.initialize()
                rag_logger.info(f"IntelligenceService initialized successfully for streaming query: {query[:50]}...")
            except Exception as init_error:
                rag_logger.error(f"IntelligenceService initialization failed: {str(init_error)}")
                rag_logger.error(f"Initialization exception type: {type(init_error).__name__}")
                import traceback
                rag_logger.error(f"Initialization full traceback: {traceback.format_exc()}")

                # Yield error message and return
                yield f"data: {json.dumps({'type': 'error', 'message': 'IntelligenceService initialization failed'})}\n\n"
                return

            # Stream the response using IntelligenceService's streaming capability
            async for chunk in intelligence_service.stream_response(
                user_query=query,
                context_chunks=sources,
                session_id=session_id
            ):
                yield chunk

        except ImportError as e:
            from backend.utils.logger import rag_logger
            import json
            rag_logger.error(f"IntelligenceService import failed with error: {str(e)}")
            rag_logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            rag_logger.error(f"Full traceback: {traceback.format_exc()}")

            # Try to identify the specific missing module
            try:
                import agents
                rag_logger.info("agents module is available")
            except ImportError as agent_import_error:
                rag_logger.error(f"agents module import failed: {str(agent_import_error)}")

            try:
                from agents_sdk.services.intelligence_service import IntelligenceService
                rag_logger.info("IntelligenceService import worked separately")
            except ImportError as intelligence_import_error:
                rag_logger.error(f"Direct IntelligenceService import failed: {str(intelligence_import_error)}")

            rag_logger.warning("IntelligenceService not available, using basic response generation")

            # Yield sources first if available
            if sources:
                for source in sources:
                    source_data = {
                        "type": "source",
                        "chunk_id": source.chunk_id,
                        "document_id": source.document_id,
                        "text": source.text[:200] + "..." if len(source.text) > 200 else source.text,
                        "score": source.score,
                        "metadata": source.metadata
                    }
                    yield f"data: {json.dumps(source_data)}\n\n"

            # Yield a fallback response
            if len(sources) > 0:
                fallback_response = f"Based on the provided context: {sources[0].text[:200] if sources[0].text else ''}... I can help answer questions about this topic. Please ask your specific question about this content."
            else:
                fallback_response = "I couldn't find any relevant information to answer your question."

            yield f"data: {json.dumps({'type': 'token', 'content': fallback_response})}\n\n"
            yield f"data: {json.dumps({'type': 'complete', 'message': 'Response completed'})}\n\n"
        except Exception as e:
            from backend.utils.logger import rag_logger
            import json
            rag_logger.error(f"Error in IntelligenceService streaming: {str(e)}")
            rag_logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            rag_logger.error(f"Full traceback: {traceback.format_exc()}")

            # Yield error message
            yield f"data: {json.dumps({'type': 'error', 'message': f'Error in IntelligenceService streaming: {str(e)}'})}\n\n"
            yield f"data: {json.dumps({'type': 'complete', 'message': 'Response completed'})}\n\n"