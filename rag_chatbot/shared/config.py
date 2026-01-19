from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional
from pydantic import Field, field_validator, computed_field
import os


class QdrantSettings(BaseSettings):
    """Settings for Qdrant vector database"""
    host: str = Field(default="http://localhost:6333", description="Qdrant host URL")
    api_key: Optional[str] = Field(default=None, description="Qdrant API key")
    collection_name: str = Field(default="book_embeddings", description="Qdrant collection name")
    vector_size: int = Field(default=1024, description="Size of embedding vectors")
    distance: str = Field(default="Cosine", description="Distance metric for similarity search")

    model_config = SettingsConfigDict(env_prefix="QDRANT_")


class NeonSettings(BaseSettings):
    """Settings for Neon Postgres database"""
    database_url: str = Field(default="postgresql://user:password@localhost:5432/dbname", description="Neon database URL")
    pool_size: int = Field(default=10, description="Connection pool size")
    pool_timeout: int = Field(default=30, description="Connection pool timeout in seconds")

    model_config = SettingsConfigDict(env_prefix="NEON_")


class GeminiSettings(BaseSettings):
    """Settings for Google Gemini embedding service"""
    api_key: str = Field(default="", description="Gemini API key")
    model: str = Field(default="gemini-embedding-001", description="Gemini embedding model")
    task_type: str = Field(default="RETRIEVAL_DOCUMENT", description="Task type for embeddings")
    output_dimensionality: int = Field(default=1024, description="Output dimensionality for embeddings")

    model_config = SettingsConfigDict(env_prefix="GEMINI_")


class LLMSettings(BaseSettings):
    """Settings for LLM provider (OpenRouter)"""
    api_key: str = Field(default="", description="OpenRouter API key")
    model: str = Field(default="openai/gpt-4-turbo", description="LLM model to use")
    base_url: str = Field(default="https://openrouter.ai/api/v1", description="OpenRouter API base URL")

    model_config = SettingsConfigDict(env_prefix="OPENROUTER_")


class CORSSettings(BaseSettings):
    """Settings for CORS configuration"""
    cors_allowed_origins: str = Field(default="http://localhost:3000,http://localhost:8000", description="CORS allowed origins")

    model_config = SettingsConfigDict(env_prefix="CORS_")

    @property
    def allowed_origins(self) -> List[str]:
        """Parse allowed_origins from the comma-separated string."""
        if hasattr(self, 'cors_allowed_origins') and self.cors_allowed_origins:
            origins = [origin.strip() for origin in self.cors_allowed_origins.split(",")]
            # If "*" is in the list, return wildcard to allow all origins
            if "*" in origins:
                return ["*"]
            return origins
        return ["*"]  # Allow all origins by default for Hugging Face deployment


class AppSettings(BaseSettings):
    """Main application settings"""
    # General settings
    app_name: str = Field(default="RAG Chatbot", description="Name of the application")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # RAG-specific settings
    max_chunk_size: int = Field(default=1200, description="Maximum size of text chunks in tokens")
    min_chunk_size: int = Field(default=800, description="Minimum size of text chunks in tokens")
    retrieval_top_k: int = Field(default=5, description="Number of top results to retrieve")
    retrieval_threshold: float = Field(default=0.7, description="Similarity threshold for retrieval")

    # Qdrant settings
    qdrant_host: str = Field(default="http://localhost:6333", description="Qdrant host URL")
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant API key")
    qdrant_collection_name: str = Field(default="book_embeddings", description="Qdrant collection name")
    qdrant_vector_size: int = Field(default=1536, description="Size of embedding vectors")
    qdrant_distance: str = Field(default="Cosine", description="Distance metric for similarity search")

    # Neon settings
    neon_database_url: str = Field(default="postgresql://user:password@localhost:5432/dbname", description="Neon database URL")
    neon_pool_size: int = Field(default=10, description="Connection pool size")
    neon_pool_timeout: int = Field(default=30, description="Connection pool timeout in seconds")

    # Gemini settings
    gemini_api_key: str = Field(default="", description="Gemini API key")
    gemini_model: str = Field(default="gemini-embedding-001", description="Gemini embedding model")
    gemini_task_type: str = Field(default="RETRIEVAL_DOCUMENT", description="Task type for embeddings")
    gemini_output_dimensionality: int = Field(default=1536, description="Output dimensionality for embeddings")

    # LLM settings (OpenRouter)
    openrouter_api_key: str = Field(default="", description="OpenRouter API key")
    openrouter_model: str = Field(default="meta-llama/llama-3.3-70b-instruct:free", description="LLM model to use")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1/chat/completions", description="OpenRouter API base URL")

    # CORS settings
    cors_cors_allowed_origins: str = Field(default="*", description="CORS allowed origins")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    @property
    def qdrant_settings(self) -> QdrantSettings:
        return QdrantSettings(
            host=self.qdrant_host,
            api_key=self.qdrant_api_key,
            collection_name=self.qdrant_collection_name,
            vector_size=self.qdrant_vector_size,
            distance=self.qdrant_distance
        )

    @property
    def neon_settings(self) -> NeonSettings:
        return NeonSettings(
            database_url=self.neon_database_url,
            pool_size=self.neon_pool_size,
            pool_timeout=self.neon_pool_timeout
        )

    @property
    def gemini_settings(self) -> GeminiSettings:
        return GeminiSettings(
            api_key=self.gemini_api_key,
            model=self.gemini_model,
            task_type=self.gemini_task_type,
            output_dimensionality=self.gemini_output_dimensionality
        )

    @property
    def llm_settings(self) -> LLMSettings:
        return LLMSettings(
            api_key=self.openrouter_api_key,
            model=self.openrouter_model,
            base_url=self.openrouter_base_url
        )

    @property
    def cors_settings(self) -> CORSSettings:
        return CORSSettings(
            cors_allowed_origins=self.cors_cors_allowed_origins
        )


# Global settings instance
settings = AppSettings()