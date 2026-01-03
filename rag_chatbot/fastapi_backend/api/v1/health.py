from fastapi import APIRouter
from typing import Dict, Any
from ...services.health_service import HealthService
from ...models.response_models import HealthResponse, ConfigResponse
from ...config import settings
from datetime import datetime

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint to verify system status.

    Returns:
        HealthResponse with system status
    """
    health_service = HealthService()
    return await health_service.get_health_status()


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check endpoint to verify if the service is ready to accept requests.

    Returns:
        Dict with readiness status
    """
    health_service = HealthService()
    is_ready = await health_service.readiness_check()
    return {
        "ready": is_ready,
        "timestamp": datetime.utcnow().isoformat(),
        "service": "RAG Chatbot API"
    }


@router.get("/status")
async def status_check() -> Dict[str, Any]:
    """
    Status check endpoint with detailed system information.

    Returns:
        Dict with detailed status information
    """
    health_service = HealthService()
    detailed_health = await health_service.get_detailed_health()
    return detailed_health


@router.get("/config", response_model=ConfigResponse)
async def get_config() -> ConfigResponse:
    """
    Configuration endpoint to return system metadata for frontend.

    Returns:
        ConfigResponse with system configuration
    """
    return ConfigResponse(
        version="v1",
        streaming_enabled=True,
        max_context_chunks=settings.max_context_chunks,
        features={
            "chat": True,
            "ingestion": True,
            "search": True,
            "streaming": True,
            "authentication": bool(settings.api_key)
        }
    )