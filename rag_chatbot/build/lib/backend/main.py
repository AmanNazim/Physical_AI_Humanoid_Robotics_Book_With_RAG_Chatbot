from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import base, health
from ..shared.config import settings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        version="1.0.0"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        # Additional security settings can be added here
    )

    # Include main routes
    app.include_router(base.router, prefix="/api/v1", tags=["base"])
    app.include_router(health.router, prefix="/api/v1", tags=["health"])

    # Placeholder routes for future implementation
    # These will be implemented in the respective modules
    @app.get("/")
    async def root():
        return {"message": "RAG Chatbot API is running"}

    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )