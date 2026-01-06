"""
Health and configuration endpoints for the RAG Chatbot API.
"""
from fastapi import APIRouter
from datetime import datetime
from typing import Dict, Any
import sys
import os

# Add the project root to the Python path to allow absolute imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.config import settings


router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint to verify system status.

    Returns:
        Dict with status and timestamp
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "RAG Chatbot API",
        "version": "1.0.0"
    }


@router.get("/config")
async def config_check() -> Dict[str, Any]:
    """
    Configuration endpoint to provide safe frontend configuration.

    Returns:
        Dict with configuration information
    """
    return {
        "feature_flags": {},
        "streaming_enabled": True,
        "ui_hints": {
            "max_context_chunks": 5,  # Using a default value since not in shared config
            "retrieval_top_k": settings.retrieval_top_k
        },
        "version": "1.0.0",
        "service": "RAG Chatbot API"
    }