"""
Embedding ingestion endpoints for the RAG Chatbot API.
"""
from fastapi import APIRouter
from typing import Dict, Any
from ..services.embedding_service import EmbeddingService
from ..schemas.embedding import EmbeddingRequest, EmbeddingResponse
from ..utils.logger import rag_logger


router = APIRouter()


@router.post("/embed")
async def embed_endpoint(request: EmbeddingRequest) -> Dict[str, Any]:
    """
    Trigger embedding ingestion workflow.
    Accept text or document payload and call the embeddings subsystem.
    """
    try:
        embedding_service = EmbeddingService()
        await embedding_service.initialize()

        result = await embedding_service.trigger_ingestion(
            text=request.text,
            document_metadata=request.document_metadata or {}
        )

        rag_logger.info(f"Embedding ingestion completed: {result['status']}")
        return result

    except Exception as e:
        rag_logger.error(f"Embedding endpoint error: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }