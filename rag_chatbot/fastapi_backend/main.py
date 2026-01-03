from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from .config import settings
from .middleware import setup_cors, setup_logging, add_exception_handlers
from .api.v1 import health_router, ingestion_router, query_router, documents_router
from .api.ws import ws_router
from .utils.logging import get_logger
from .databases.database_manager import initialize_database_manager, shutdown_database_manager

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    """
    # Startup
    logger.info("Starting up FastAPI application...")
    try:
        # Initialize database connections
        await initialize_database_manager()
        logger.info("Database connections initialized")
    except Exception as e:
        logger.error(f"Error initializing database connections: {str(e)}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down FastAPI application...")
    try:
        # Close database connections
        await shutdown_database_manager()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error shutting down database connections: {str(e)}")
        raise


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        version="1.0.0",
        lifespan=lifespan
    )

    # Setup middleware
    setup_cors(app)
    setup_logging(app)
    add_exception_handlers(app)

    # Include API routers with version prefix
    app.include_router(health_router, prefix=settings.api_v1_prefix, tags=["health"])
    app.include_router(ingestion_router, prefix=settings.api_v1_prefix, tags=["ingestion"])
    app.include_router(query_router, prefix=settings.api_v1_prefix, tags=["query"])
    app.include_router(documents_router, prefix=settings.api_v1_prefix, tags=["documents"])
    app.include_router(ws_router, prefix=settings.api_v1_prefix, tags=["websocket"])

    # Root endpoint
    @app.get("/")
    async def root():
        return {"message": "RAG Chatbot FastAPI Backend is running"}

    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )