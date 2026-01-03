import uuid
import time
import json
from typing import Callable, Awaitable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from ..utils.logging import get_logger

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Custom logging middleware that adds structured logging with request IDs and timing.
    """
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        # Generate a unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Log request start
        start_time = time.time()
        logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
            }
        )

        try:
            # Process the request
            response = await call_next(request)
        except Exception as e:
            # Log the error
            duration = time.time() - start_time
            logger.error(
                f"Request failed: {str(e)}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "duration": duration,
                    "error": str(e),
                }
            )
            # Re-raise the exception
            raise

        # Calculate duration
        duration = time.time() - start_time

        # Log request completion
        logger.info(
            "Request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "status_code": response.status_code,
                "duration": duration,
            }
        )

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response


def setup_logging(app) -> None:
    """
    Add logging middleware to the application.

    Args:
        app: FastAPI application instance
    """
    app.add_middleware(LoggingMiddleware)