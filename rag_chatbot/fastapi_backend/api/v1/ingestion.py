from fastapi import APIRouter
from ...services.ingestion_service import IngestionService
from ...models.request_models import IngestTextRequest
from ...models.response_models import IngestionResponse

router = APIRouter(tags=["ingestion"])


@router.post("/embed-text", response_model=IngestionResponse)
async def embed_text(request: IngestTextRequest) -> IngestionResponse:
    """
    Endpoint to send raw text to embeddings engine.

    Args:
        request: IngestTextRequest containing the text and metadata

    Returns:
        IngestionResponse with the ingestion results
    """
    ingestion_service = IngestionService()
    await ingestion_service.initialize()
    return await ingestion_service.ingest_text(request)


@router.post("/add-document", response_model=IngestionResponse)
async def add_document(request: IngestTextRequest) -> IngestionResponse:
    """
    Endpoint for chunk → embed → store workflow.

    Args:
        request: IngestTextRequest containing the document content and metadata

    Returns:
        IngestionResponse with the ingestion results
    """
    ingestion_service = IngestionService()
    await ingestion_service.initialize()
    return await ingestion_service.ingest_text(request)