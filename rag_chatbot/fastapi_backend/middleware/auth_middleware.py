from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from ..config import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)
security = HTTPBearer()


class APIKeyValidator:
    """
    API key validation utility class.
    """
    @staticmethod
    def validate_api_key(provided_key: str) -> bool:
        """
        Validate the provided API key against the configured key.

        Args:
            provided_key: The API key provided by the client

        Returns:
            bool: True if the key is valid
        """
        if not settings.api_key:
            # If no API key is configured, allow all requests
            return True

        if not provided_key:
            return False

        # Compare the provided key with the configured key
        return provided_key == settings.api_key


async def get_api_key_auth(request: Request) -> bool:
    """
    Dependency to validate API key from request.

    Args:
        request: FastAPI request object

    Returns:
        bool: True if API key is valid
    """
    # Try to get API key from header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        provided_key = auth_header[7:]  # Remove "Bearer " prefix
    elif auth_header and auth_header.startswith("ApiKey "):
        provided_key = auth_header[7:]  # Remove "ApiKey " prefix
    else:
        # Try to get API key from query parameter
        provided_key = request.query_params.get("api_key")

    # Validate the API key
    is_valid = APIKeyValidator.validate_api_key(provided_key)

    if not is_valid:
        logger.warning(f"Invalid API key attempt from {request.client.host}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    return is_valid


def require_api_key(request: Request) -> bool:
    """
    Function to require API key for specific endpoints.

    Args:
        request: FastAPI request object

    Returns:
        bool: True if API key is valid
    """
    return get_api_key_auth(request)