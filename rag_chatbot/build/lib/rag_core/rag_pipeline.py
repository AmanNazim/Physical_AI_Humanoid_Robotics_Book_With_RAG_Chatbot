from typing import List, Dict, Any, Optional
from .interfaces.retriever_interface import RetrieverInterface, RetrievalResult
from .interfaces.agent_interface import AgentInterface, AgentResponse, AgentContext
from .utils.logger import rag_logger


class RAGPipeline:
    """
    High-level RAG flow contract.
    Defines the main pipeline for retrieval-augmented generation without implementation logic.
    """

    def __init__(
        self,
        retriever: RetrieverInterface,
        agent: AgentInterface
    ):
        """
        Initialize the RAG pipeline with required components.

        Args:
            retriever: Retriever component for context retrieval
            agent: Agent component for answer generation
        """
        self.retriever = retriever
        self.agent = agent

    async def retrieve_chunks(self, query: str) -> List[RetrievalResult]:
        """
        Retrieve relevant text chunks based on the query.

        Args:
            query: User query for retrieval

        Returns:
            List of retrieved chunks with metadata
        """
        rag_logger.info("Starting chunk retrieval", extra_data={"query": query})
        chunks = await self.retriever.retrieve_chunks(query)
        rag_logger.info(
            f"Retrieved {len(chunks)} chunks",
            extra_data={"query": query, "chunk_count": len(chunks)}
        )
        return chunks

    async def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        mode: str = "full_book"
    ) -> AgentResponse:
        """
        Generate an answer based on the query and context chunks.

        Args:
            query: User query
            context_chunks: Retrieved context chunks
            mode: Retrieval mode ('full_book' or 'selected_text_only')

        Returns:
            Generated answer with citations
        """
        rag_logger.info(
            "Starting answer generation",
            extra_data={"query": query, "mode": mode, "context_chunk_count": len(context_chunks)}
        )

        # Create agent context
        context = AgentContext(
            query=query,
            retrieved_chunks=context_chunks,
            mode=mode
        )

        # Generate response using agent
        response = await self.agent.process_query(context)

        rag_logger.info(
            "Answer generation completed",
            extra_data={
                "query": query,
                "response_length": len(response.response),
                "citation_count": len(response.citations)
            }
        )
        return response

    async def format_response(
        self,
        answer: str,
        citations: List[str]
    ) -> Dict[str, Any]:
        """
        Format the response with proper structure and citations.

        Args:
            answer: Generated answer text
            citations: List of citations for the answer

        Returns:
            Formatted response with proper structure
        """
        rag_logger.info(
            "Formatting response",
            extra_data={"answer_length": len(answer), "citation_count": len(citations)}
        )

        formatted_response = {
            "answer": answer,
            "citations": citations,
            "formatted_at": "timestamp_placeholder",
            "response_format_version": "1.0"
        }

        rag_logger.info("Response formatting completed")
        return formatted_response

    async def run_rag_pipeline(
        self,
        query: str,
        mode: str = "full_book",
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Execute the complete RAG pipeline from query to response.

        Args:
            query: User query
            mode: Retrieval mode ('full_book' or 'selected_text_only')
            top_k: Number of top results to retrieve

        Returns:
            Complete response with answer and metadata
        """
        rag_logger.info(
            "Starting RAG pipeline execution",
            extra_data={"query": query, "mode": mode, "top_k": top_k}
        )

        try:
            # Step 1: Retrieve relevant chunks
            retrieval_results = await self.retriever.retrieve_chunks(
                query=query,
                top_k=top_k
            )

            # Convert retrieval results to context format
            context_chunks = [
                {
                    "chunk_id": result.chunk_id,
                    "text": result.text_content,
                    "document_reference": result.document_reference,
                    "score": result.score,
                    "metadata": result.metadata
                }
                for result in retrieval_results
            ]

            # Step 2: Generate answer
            agent_response = await self.generate_answer(
                query=query,
                context_chunks=context_chunks,
                mode=mode
            )

            # Step 3: Format response
            formatted_response = await self.format_response(
                answer=agent_response.response,
                citations=agent_response.citations
            )

            # Add additional metadata to response
            formatted_response.update({
                "query": query,
                "mode": mode,
                "retrieved_chunks_count": len(context_chunks),
                "pipeline_execution_successful": True
            })

            rag_logger.info("RAG pipeline execution completed successfully")
            return formatted_response

        except Exception as e:
            rag_logger.error(
                f"RAG pipeline execution failed: {str(e)}",
                extra_data={"query": query, "mode": mode, "error": str(e)}
            )
            # Return error response
            return {
                "answer": "An error occurred while processing your request.",
                "citations": [],
                "query": query,
                "mode": mode,
                "pipeline_execution_successful": False,
                "error": str(e)
            }