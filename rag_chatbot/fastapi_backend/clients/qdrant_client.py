from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient as QdrantBaseClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from ..config import settings
from ..utils.logging import get_logger
from ..models.response_models import Source
import uuid

logger = get_logger(__name__)


class QdrantClient:
    """
    Qdrant client wrapper for the FastAPI backend.
    This client provides a clean interface to interact with the Qdrant vector database.
    """

    def __init__(self):
        # Initialize Qdrant client with settings
        self.client = QdrantBaseClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            prefer_grpc=False  # Using HTTP for better compatibility
        )
        self.collection_name = settings.qdrant_collection_name
        self.vector_size = settings.qdrant_vector_size

    async def ensure_collection_exists(self) -> bool:
        """
        Ensure that the required collection exists in Qdrant.

        Returns:
            bool: True if collection exists or was created successfully
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                # Create collection with specified vector size
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Qdrant collection exists: {self.collection_name}")

            return True
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
            return False

    async def insert_vectors(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> bool:
        """
        Insert vectors into the Qdrant collection.

        Args:
            vectors: List of embedding vectors to insert
            payloads: List of metadata payloads for each vector
            ids: Optional list of IDs for the vectors (auto-generated if not provided)

        Returns:
            bool: True if insertion was successful
        """
        try:
            if ids is None:
                # Generate UUIDs if not provided
                ids = [str(uuid.uuid4()) for _ in range(len(vectors))]

            # Prepare points for insertion
            points = []
            for i, (vector, payload) in enumerate(zip(vectors, payloads)):
                points.append(
                    models.PointStruct(
                        id=ids[i],
                        vector=vector,
                        payload=payload
                    )
                )

            # Insert points into collection
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            logger.info(f"Inserted {len(vectors)} vectors into Qdrant collection: {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"Error inserting vectors into Qdrant: {str(e)}")
            return False

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Source]:
        """
        Search for similar vectors in the Qdrant collection.

        Args:
            query_vector: The query embedding vector
            top_k: Number of top results to return
            filters: Optional filters to apply to the search

        Returns:
            List of Source objects containing the search results
        """
        try:
            # Prepare filters if provided
            qdrant_filters = None
            if filters:
                # Convert filters to Qdrant filter format
                must_conditions = []
                for key, value in filters.items():
                    if isinstance(value, str):
                        must_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=value)
                            )
                        )
                    elif isinstance(value, (int, float)):
                        must_conditions.append(
                            models.FieldCondition(
                                key=key,
                                range=models.Range(gte=value, lte=value)
                            )
                        )

                if must_conditions:
                    qdrant_filters = models.Filter(must=must_conditions)

            # Perform search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=qdrant_filters,
                with_payload=True,
                with_vectors=False
            )

            # Convert results to Source objects
            sources = []
            for result in search_results:
                source = Source(
                    chunk_id=str(result.id),
                    document_id=result.payload.get("document_id", ""),
                    text=result.payload.get("text", ""),
                    score=result.score,
                    metadata=result.payload.get("metadata", {})
                )
                sources.append(source)

            logger.info(f"Found {len(sources)} results from Qdrant search")
            return sources

        except Exception as e:
            logger.error(f"Error searching in Qdrant: {str(e)}")
            return []

    async def delete_vectors(self, ids: List[str]) -> bool:
        """
        Delete vectors from the Qdrant collection by their IDs.

        Args:
            ids: List of vector IDs to delete

        Returns:
            bool: True if deletion was successful
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=ids
                )
            )

            logger.info(f"Deleted {len(ids)} vectors from Qdrant collection: {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"Error deleting vectors from Qdrant: {str(e)}")
            return False

    async def get_vector_count(self) -> int:
        """
        Get the total count of vectors in the collection.

        Returns:
            int: Number of vectors in the collection
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception as e:
            logger.error(f"Error getting vector count from Qdrant: {str(e)}")
            return 0