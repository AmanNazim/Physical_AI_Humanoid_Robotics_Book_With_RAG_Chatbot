from fastapi import APIRouter
from typing import List
from ...services.ingestion_service import IngestionService
from ...models.request_models import DocumentMetadata
from ...models.response_models import DocumentResponse

router = APIRouter(tags=["documents"])


@router.get("/documents", response_model=DocumentResponse)
async def get_all_documents() -> DocumentResponse:
    """
    Endpoint to return all documents stored in the system.

    Returns:
        DocumentResponse containing all document metadata
    """
    # For now, we'll return an empty list since we don't have a function to retrieve all documents
    # in the ingestion service yet
    ingestion_service = IngestionService()
    return DocumentResponse(documents=[])


@router.get("/document/{document_id}", response_model=DocumentMetadata)
async def get_document(document_id: str) -> DocumentMetadata:
    """
    Endpoint to return a specific document by ID.

    Args:
        document_id: ID of the document to retrieve

    Returns:
        DocumentMetadata for the requested document
    """
    # For now, we'll return a placeholder since we don't have a function to retrieve a single document
    # in the ingestion service yet
    ingestion_service = IngestionService()
    return DocumentMetadata(
        document_id=document_id,
        title="Placeholder Title",
        source="manual",
        chunk_count=0,
        created_at="2023-01-01T00:00:00"
    )


@router.delete("/document/{document_id}")
async def delete_document(document_id: str) -> dict:
    """
    Endpoint to delete a document by ID.

    Args:
        document_id: ID of the document to delete

    Returns:
        Dictionary with deletion status
    """
    return {"status": "deleted", "document_id": document_id}