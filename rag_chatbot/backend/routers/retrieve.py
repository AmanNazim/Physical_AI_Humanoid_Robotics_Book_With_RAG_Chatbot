"""
Retrieval endpoints for the RAG Chatbot API.
"""
from fastapi import APIRouter
from typing import Dict, Any
from ..services.retrieval_service import RetrievalService
from ..schemas.retrieval import RetrievalRequest, RetrievalResponse
from ..utils.logger import rag_logger


router = APIRouter()


@router.post("/retrieve")
async def retrieve_endpoint(request: RetrievalRequest) -> Dict[str, Any]:
    """
    Pure retrieval endpoint (no LLM processing).
    Accept query and call Qdrant via Database Subsystem, return top-k chunks + metadata.
    """
    try:
        retrieval_service = RetrievalService()
        await retrieval_service.initialize()

        sources = await retrieval_service.retrieve_by_query(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters
        )

        result = {
            "sources": [source.model_dump() for source in sources],
            "query": request.query,
            "retrieved_count": len(sources)
        }

        rag_logger.info(f"Retrieved {len(sources)} results for query: {request.query[:50]}...")
        return result

    except Exception as e:
        rag_logger.error(f"Retrieve endpoint error: {str(e)}")
        return {
            "sources": [],
            "query": request.query,
            "retrieved_count": 0,
            "error": str(e)
        }