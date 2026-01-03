from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from ..models.error_models import ErrorResponse, ErrorDetail, ErrorCode
from ..utils.logging import get_logger
import traceback

logger = get_logger(__name__)


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    logger.error(f"Validation error: {exc.errors()}")

    error_detail = ErrorDetail(
        code=ErrorCode.VALIDATION_ERROR,
        message="Request validation failed",
        details={"errors": exc.errors()}
    )

    return JSONResponse(
        status_code=422,
        content=ErrorResponse(error=error_detail).model_dump()
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions"""
    logger.error(f"HTTP error {exc.status_code}: {exc.detail}")

    # Map HTTP status codes to error codes
    error_code_map = {
        400: ErrorCode.VALIDATION_ERROR,
        401: ErrorCode.AUTHENTICATION_ERROR,
        403: ErrorCode.AUTHORIZATION_ERROR,
        404: ErrorCode.NOT_FOUND,
        429: ErrorCode.RATE_LIMIT_EXCEEDED,
        500: ErrorCode.INTERNAL_ERROR,
        503: ErrorCode.SERVICE_UNAVAILABLE,
    }

    error_code = error_code_map.get(exc.status_code, ErrorCode.INTERNAL_ERROR)

    error_detail = ErrorDetail(
        code=error_code,
        message=exc.detail if isinstance(exc.detail, str) else str(exc.detail),
        details={"status_code": exc.status_code}
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=error_detail).model_dump()
    )


async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"General error: {str(exc)}")
    logger.error(f"Traceback: {traceback.format_exc()}")

    error_detail = ErrorDetail(
        code=ErrorCode.INTERNAL_ERROR,
        message="An internal server error occurred",
        details={"error_type": type(exc).__name__, "error_message": str(exc)}
    )

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error=error_detail).model_dump()
    )


def add_exception_handlers(app: FastAPI) -> None:
    """
    Add global exception handlers to the application.

    Args:
        app: FastAPI application instance
    """
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)