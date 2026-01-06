"""
Middleware package for the RAG Chatbot backend.
"""
from fastapi import FastAPI
from .cors import add_cors_middleware
from .logging import add_logging_middleware
from .rate_limit import add_rate_limit_middleware


def setup_middleware(app: FastAPI, requests_per_minute: int = 60):
    """
    Setup all middleware for the FastAPI application.

    Args:
        app: FastAPI application instance
        requests_per_minute: Number of requests allowed per minute per IP for rate limiting
    """
    # Add CORS middleware first
    add_cors_middleware(app)

    # Add logging middleware
    add_logging_middleware(app)

    # Add rate limiting middleware last (so it can access request state set by logging middleware)
    add_rate_limit_middleware(app, requests_per_minute=requests_per_minute)