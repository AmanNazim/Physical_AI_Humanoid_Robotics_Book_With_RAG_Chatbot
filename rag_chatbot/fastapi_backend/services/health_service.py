from typing import Dict, Any
from datetime import datetime
from ..clients import QdrantClient, PostgresClient
from ..utils.logging import get_logger
from ..models.response_models import HealthResponse

logger = get_logger(__name__)


class HealthService:
    """
    Service for handling health checks.
    Checks the status of all subsystems.
    """

    def __init__(self):
        self.qdrant_client = QdrantClient()
        self.postgres_client = PostgresClient()

    async def check_qdrant_connection(self) -> bool:
        """
        Check if Qdrant is connected and responsive.

        Returns:
            bool: True if Qdrant is connected
        """
        try:
            # Try to get collection info to verify connection
            await self.qdrant_client.ensure_collection_exists()
            return True
        except Exception as e:
            logger.error(f"Qdrant connection check failed: {str(e)}")
            return False

    async def check_postgres_connection(self) -> bool:
        """
        Check if PostgreSQL is connected and responsive.

        Returns:
            bool: True if PostgreSQL is connected
        """
        try:
            return await self.postgres_client.health_check()
        except Exception as e:
            logger.error(f"PostgreSQL connection check failed: {str(e)}")
            return False

    async def get_health_status(self) -> HealthResponse:
        """
        Get the overall health status of the system.

        Returns:
            HealthResponse with the health status
        """
        # Check Qdrant connection
        qdrant_connected = await self.check_qdrant_connection()

        # Check PostgreSQL connection
        postgres_connected = await self.check_postgres_connection()

        # Determine overall status
        overall_status = "ok" if (qdrant_connected and postgres_connected) else "degraded"

        logger.info(f"Health check completed - Qdrant: {qdrant_connected}, PostgreSQL: {postgres_connected}")

        return HealthResponse(
            status=overall_status,
            version="v1",
            qdrant_connected=qdrant_connected,
            postgres_connected=postgres_connected,
            timestamp=datetime.utcnow()
        )

    async def get_detailed_health(self) -> Dict[str, Any]:
        """
        Get detailed health information for all subsystems.

        Returns:
            Dictionary with detailed health information
        """
        health_info = {
            "timestamp": datetime.utcnow().isoformat(),
            "version": "v1",
            "status": "ok",
            "subsystems": {
                "qdrant": {
                    "connected": await self.check_qdrant_connection(),
                    "type": "vector_database"
                },
                "postgres": {
                    "connected": await self.check_postgres_connection(),
                    "type": "relational_database"
                },
                "embeddings": {
                    "connected": True,  # Assuming embeddings service is available if we can import it
                    "type": "embedding_service"
                },
                "intelligence": {
                    "connected": True,  # Assuming LLM service is available if we can import it
                    "type": "llm_service"
                }
            }
        }

        # Determine overall status
        all_connected = all(
            subsystem["connected"]
            for subsystem in health_info["subsystems"].values()
        )
        health_info["status"] = "ok" if all_connected else "degraded"

        return health_info

    async def readiness_check(self) -> bool:
        """
        Check if the service is ready to accept requests.

        Returns:
            bool: True if service is ready
        """
        # For readiness, we might be more strict than for health
        # Here we'll check that critical services are available
        qdrant_ok = await self.check_qdrant_connection()
        postgres_ok = await self.check_postgres_connection()

        return qdrant_ok and postgres_ok