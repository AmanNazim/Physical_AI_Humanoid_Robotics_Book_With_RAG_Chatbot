from pydantic_settings import BaseSettings
from typing import List, Optional
from pydantic import Field
import os


class Settings(BaseSettings):
    """Application settings"""

    # General settings
    app_name: str = Field(default="RAG Chatbot FastAPI Backend", description="Name of the application")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    service_name: str = Field(default="fastapi-backend", description="Name of this service for logging")

    # API settings
    api_v1_prefix: str = Field(default="/api/v1", description="API version 1 prefix")

    # CORS settings
    allowed_cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:8000",
        description="Comma-separated list of allowed CORS origins"
    )

    # Database settings (from shared config)
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant host URL")
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant API key")
    qdrant_collection_name: str = Field(default="book_embeddings", description="Qdrant collection name")
    qdrant_vector_size: int = Field(default=1536, description="Size of embedding vectors")

    neon_postgres_url: str = Field(
        default="postgresql://user:password@localhost:5432/dbname",
        description="Neon database URL"
    )

    # Embedding settings
    gemini_api_key: str = Field(default="", description="Gemini API key")
    gemini_model: str = Field(default="gemini-embedding-001", description="Gemini embedding model")
    embedding_dimension: int = Field(default=1536, description="Output dimensionality for embeddings")

    # LLM settings
    llm_api_key: str = Field(default="", description="LLM API key")
    llm_model: str = Field(default="openai/gpt-4-turbo", description="LLM model to use")
    llm_base_url: str = Field(default="https://openrouter.ai/api/v1", description="LLM API base URL")

    # RAG settings
    max_context_chunks: int = Field(default=5, description="Maximum number of context chunks to retrieve")
    retrieval_top_k: int = Field(default=5, description="Number of top results to retrieve")
    retrieval_threshold: float = Field(default=0.7, description="Similarity threshold for retrieval")

    # Security settings
    fastapi_secret_key: str = Field(default="your-secret-key-here", description="Secret key for security")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse allowed_cors_origins from the comma-separated string."""
        if self.allowed_cors_origins:
            return [origin.strip() for origin in self.allowed_cors_origins.split(",")]
        return ["http://localhost:3000", "http://localhost:8000"]


# Global settings instance
settings = Settings()