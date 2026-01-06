"""
Logging middleware for the RAG Chatbot API.
"""
import time
import uuid
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from ..utils.logger import rag_logger
import json


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Custom logging middleware to log request and response details.
    """

    def __init__(self, app: FastAPI):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        # Generate a unique request ID for tracking
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Log request details
        start_time = time.time()
        rag_logger.info(
            f"REQUEST_START - ID: {request_id} | "
            f"Method: {request.method} | "
            f"Path: {request.url.path} | "
            f"Query: {request.url.query} | "
            f"Client: {request.client.host}:{request.client.port} | "
            f"Headers: {dict(request.headers)}"
        )

        try:
            # Process the request
            response: Response = await call_next(request)

            # Calculate processing time
            process_time = time.time() - start_time

            # Log response details
            rag_logger.info(
                f"REQUEST_END - ID: {request_id} | "
                f"Status: {response.status_code} | "
                f"ProcessTime: {process_time:.3f}s | "
                f"Content-Length: {response.headers.get('content-length', 'unknown')}"
            )

            # Add request ID to response headers for client tracking
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.3f}s"

            return response

        except Exception as e:
            # Calculate processing time for error case
            process_time = time.time() - start_time

            # Log error details
            rag_logger.error(
                f"REQUEST_ERROR - ID: {request_id} | "
                f"Method: {request.method} | "
                f"Path: {request.url.path} | "
                f"Error: {str(e)} | "
                f"ProcessTime: {process_time:.3f}s"
            )

            # Re-raise the exception to be handled by FastAPI's exception handlers
            raise


def add_logging_middleware(app: FastAPI):
    """
    Add logging middleware to the FastAPI application.

    Args:
        app: FastAPI application instance
    """
    app.add_middleware(LoggingMiddleware)