"""
Configuration module for the Embeddings & Chunking Pipeline.
This module defines configuration parameters for chunking, embedding, and database connectivity.
"""
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class EmbeddingConfig(BaseSettings):
    """
    Configuration class for the Embeddings & Chunking Pipeline.
    """
    # Embedding model configuration
    embed_model_name: str = Field(default="models/embedding-001", description="Embedding model name")
    embedding_dim: int = Field(default=1024, description="Dimension of generated embeddings (for Gemini embedding-001)")

    # Chunking parameters - following the specification of 800-1200 tokens
    chunk_size: int = Field(default=1000, description="Target chunk size in tokens")
    min_chunk_size: int = Field(default=800, description="Minimum chunk size in tokens")
    max_chunk_size: int = Field(default=1200, description="Maximum chunk size in tokens")
    chunk_overlap: int = Field(default=200, description="Overlap size between chunks in tokens (20% of target)")

    # Batching parameters - following specification of up to 96 chunks per batch
    max_batch_size: int = Field(default=96, description="Maximum number of chunks per API batch")

    # Gemini API configuration
    gemini_api_key: Optional[str] = Field(default=None, description="Gemini API key")
    gemini_model_name: str = Field(default="gemini-embedding-001", description="Gemini embedding model name")
    gemini_task_type: str = Field(default="RETRIEVAL_DOCUMENT", description="Task type for embeddings")
    gemini_output_dimensionality: int = Field(default=1024, description="Output dimensionality for embeddings")

    # Qdrant configuration
    qdrant_host: str = Field(default="http://localhost:6333", description="Qdrant host URL")
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant API key")
    qdrant_collection_name: str = Field(default="book_embeddings", description="Qdrant collection name")

    # Database configuration (using existing settings)
    neon_database_url: str = Field(default="postgresql://user:password@localhost:5432/dbname", description="Neon database URL")

    # Processing parameters
    max_retries: int = Field(default=3, description="Maximum number of retries for API calls")
    retry_delay_base: float = Field(default=1.0, description="Base delay for exponential backoff")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra environment variables that don't match fields
    )


# Global configuration instance
config = EmbeddingConfig()