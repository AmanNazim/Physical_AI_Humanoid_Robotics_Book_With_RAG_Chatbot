"""
Main FastAPI application for the RAG Chatbot API.
"""
from fastapi import FastAPI
import sys
import os
# Add the project root to the Python path to allow absolute imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.config import settings
from backend.middleware import setup_middleware
from backend.routers import health, chat, retrieve, embed, config
from backend.utils.logger import rag_logger
import uvicorn


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="RAG Chatbot API",
        description="Production-ready FastAPI backend for RAG system",
        version="1.0.0",
        debug=settings.debug
    )

    # Setup all middleware (CORS, logging, rate limiting)
    setup_middleware(app, requests_per_minute=60)  # Using a default value

    # Include API routers
    app.include_router(health, prefix="/api/v1", tags=["health"])
    app.include_router(chat, prefix="/api/v1", tags=["chat"])
    app.include_router(retrieve, prefix="/api/v1", tags=["retrieve"])
    app.include_router(embed, prefix="/api/v1", tags=["embed"])
    app.include_router(config, prefix="/api/v1", tags=["config"])

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "message": "RAG Chatbot API is running",
            "version": "1.0.0",
            "endpoints": [
                "/api/v1/health",
                "/api/v1/chat",
                "/api/v1/chat/stream",
                "/api/v1/retrieve",
                "/api/v1/embed",
                "/api/v1/config"
            ]
        }

    # Health check endpoint at root as well
    @app.get("/health")
    async def root_health():
        return {"status": "ok", "service": "RAG Chatbot API"}

    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    # Configure logging
    import logging
    logging.basicConfig(level=logging.INFO)

    rag_logger.info("Starting RAG Chatbot API server...")

    uvicorn.run(
        app,  # Pass the app instance directly
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower()
    )