"""
CORS middleware for the RAG Chatbot API.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# Add the project root to the Python path to allow absolute imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.config import settings


def add_cors_middleware(app: FastAPI):
    """
    Add CORS middleware to the FastAPI application.

    Args:
        app: FastAPI application instance
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        # Expose headers that clients may need to access
        expose_headers=["Content-Type", "X-Total-Count", "X-Request-ID"],
    )