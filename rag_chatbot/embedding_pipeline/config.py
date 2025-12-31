import os
from typing import Optional
from dotenv import load_dotenv

# Load environment vairables from .env file.
load_dotenv()

# Configuration variables for the embeddings pipeline
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "gemini-embedding-001")
CHUNK_SIZE_MIN = int(os.getenv("CHUNK_SIZE_MIN", "800"))
CHUNK_SIZE_TARGET = int(os.getenv("CHUNK_SIZE_TARGET", "1000"))
CHUNK_SIZE_MAX = int(os.getenv("CHUNK_SIZE_MAX", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))  # tokens
EMBEDDING_DIM = int(os.getenv("QDRANT_VECTOR_SIZE", "1536"))  # Using value from .env file
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "book_embeddings")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Safety checks ensuring all values exist
def validate_config():
    """Validate that all required configuration values are present."""
    errors = []

    if not EMBED_MODEL_NAME:
        errors.append("EMBED_MODEL_NAME is required")

    if CHUNK_SIZE_MIN <= 0:
        errors.append("CHUNK_SIZE_MIN must be positive")

    if CHUNK_SIZE_MAX < CHUNK_SIZE_MIN:
        errors.append("CHUNK_SIZE_MAX must be >= CHUNK_SIZE_MIN")

    if CHUNK_OVERLAP >= CHUNK_SIZE_MIN:
        errors.append("CHUNK_OVERLAP must be less than CHUNK_SIZE_MIN")

    if EMBEDDING_DIM <= 0:
        errors.append("EMBEDDING_DIM must be positive")

    if not QDRANT_HOST:
        errors.append("QDRANT_HOST is required")

    if QDRANT_PORT <= 0 or QDRANT_PORT > 65535:
        errors.append("QDRANT_PORT must be between 1 and 65535")

    if not QDRANT_COLLECTION_NAME:
        errors.append("QDRANT_COLLECTION_NAME is required")

    if not NEON_DATABASE_URL:
        errors.append("NEON_DATABASE_URL is required")

    if not GEMINI_API_KEY:
        errors.append("GEMINI_API_KEY is required")

    if BATCH_SIZE <= 0:
        errors.append("BATCH_SIZE must be positive")

    if EMBEDDING_DIM <= 0:
        errors.append("EMBEDDING_DIM must be positive")

    if errors:
        raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")


# Validate config on import only if all required environment variables are set
def _validate_on_import():
    """Validate config on import if all required variables are present"""
    import os
    required_vars = ['GEMINI_API_KEY', 'NEON_DATABASE_URL', 'QDRANT_API_KEY']
    all_set = all(os.getenv(var) for var in required_vars)
    if all_set:
        validate_config()

_validate_on_import()