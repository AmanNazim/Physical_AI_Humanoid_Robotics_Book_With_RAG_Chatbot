from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ..config import settings


def setup_cors(app: FastAPI) -> None:
    """
    Setup CORS middleware for the application.

    Args:
        app: FastAPI application instance
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        # Additional security settings
        allow_origin_regex=None,
        # Expose headers for client access
        expose_headers=["Access-Control-Allow-Origin"]
    )