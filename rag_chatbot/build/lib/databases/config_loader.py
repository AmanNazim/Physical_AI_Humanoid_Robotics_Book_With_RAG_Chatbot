from typing import Optional
from pydantic import BaseModel
from shared.config import settings


class DatabaseConfig(BaseModel):
    """
    Typed configuration object for database settings.
    """
    qdrant_host: str
    qdrant_api_key: Optional[str]
    qdrant_collection_name: str
    qdrant_vector_size: int
    qdrant_distance: str

    neon_database_url: str
    neon_pool_size: int
    neon_pool_timeout: int

    @classmethod
    def from_global_settings(cls) -> 'DatabaseConfig':
        """
        Create a DatabaseConfig instance from the global settings.
        """
        return cls(
            qdrant_host=settings.qdrant.host,
            qdrant_api_key=settings.qdrant.api_key,
            qdrant_collection_name=settings.qdrant.collection_name,
            qdrant_vector_size=settings.qdrant.vector_size,
            qdrant_distance=settings.qdrant.distance,

            neon_database_url=settings.neon.database_url,
            neon_pool_size=settings.neon.pool_size,
            neon_pool_timeout=settings.neon.pool_timeout
        )

    def validate(self) -> bool:
        """
        Validate that the configuration values are acceptable.
        """
        errors = []

        # Validate Qdrant settings
        if not self.qdrant_host:
            errors.append("Qdrant host is required")

        if self.qdrant_vector_size <= 0:
            errors.append("Qdrant vector size must be positive")

        if self.qdrant_distance.upper() not in ["COSINE", "EUCLID", "DOT"]:
            errors.append("Qdrant distance must be one of: COSINE, EUCLID, DOT")

        # Validate Neon settings
        if not self.neon_database_url or "postgresql" not in self.neon_database_url.lower():
            errors.append("Valid PostgreSQL database URL is required")

        if self.neon_pool_size <= 0:
            errors.append("Neon pool size must be positive")

        if self.neon_pool_timeout <= 0:
            errors.append("Neon pool timeout must be positive")

        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

        return True


class ConfigLoader:
    """
    Configuration loader for database settings.
    """

    @staticmethod
    def load_config() -> DatabaseConfig:
        """
        Load database configuration from global settings.
        """
        config = DatabaseConfig.from_global_settings()
        config.validate()
        return config

    @staticmethod
    def get_qdrant_config() -> dict:
        """
        Get Qdrant-specific configuration as a dictionary.
        """
        config = ConfigLoader.load_config()
        return {
            "host": config.qdrant_host,
            "api_key": config.qdrant_api_key,
            "collection_name": config.qdrant_collection_name,
            "vector_size": config.qdrant_vector_size,
            "distance": config.qdrant_distance
        }

    @staticmethod
    def get_neon_config() -> dict:
        """
        Get Neon-specific configuration as a dictionary.
        """
        config = ConfigLoader.load_config()
        return {
            "database_url": config.neon_database_url,
            "pool_size": config.neon_pool_size,
            "pool_timeout": config.neon_pool_timeout
        }

    @staticmethod
    def validate_environment() -> bool:
        """
        Validate that the environment has the required configuration.
        """
        try:
            config = ConfigLoader.load_config()
            return True
        except ValueError as e:
            print(f"Configuration validation error: {e}")
            return False
        except Exception as e:
            print(f"Environment validation error: {e}")
            return False


# Global instance for easy access
config_loader = ConfigLoader()


def get_database_config() -> DatabaseConfig:
    """
    Get the database configuration.
    """
    return config_loader.load_config()


def get_qdrant_config_dict() -> dict:
    """
    Get Qdrant configuration as a dictionary.
    """
    return config_loader.get_qdrant_config()


def get_neon_config_dict() -> dict:
    """
    Get Neon configuration as a dictionary.
    """
    return config_loader.get_neon_config()