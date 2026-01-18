import asyncio
from typing import Optional, List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse
from shared.config import settings
from rag_core.utils.logger import rag_logger
from rag_core.utils.timing import timing_decorator


class QdrantClientWrapper:
    """
    Wrapper for Qdrant client with robust retry logic and connection management.
    """

    def __init__(self):
        self._client: Optional[QdrantClient] = None
        self._collection_name = settings.qdrant_settings.collection_name
        self._vector_size = settings.qdrant_settings.vector_size
        self._distance = settings.qdrant_settings.distance

    async def connect(self):
        """
        Initialize the Qdrant client with connection parameters from config.
        """
        try:
            # Use the correct QdrantClient initialization for cloud instance
            self._client = QdrantClient(
                url=settings.qdrant_settings.host,
                api_key=settings.qdrant_settings.api_key,
                # Remove prefer_grpc for cloud instances which may not support it
            )
            rag_logger.info("Qdrant client initialized successfully")
        except Exception as e:
            rag_logger.error(f"Failed to initialize Qdrant client: {str(e)}")
            raise

    async def disconnect(self):
        """
        Close the Qdrant client connection.
        """
        if self._client:
            # Qdrant client doesn't have a disconnect method, but we can log the event
            rag_logger.info("Qdrant client disconnected")

    @property
    def client(self) -> QdrantClient:
        """
        Get the Qdrant client instance.
        """
        if not self._client:
            raise RuntimeError("Qdrant client not initialized. Call connect() first.")
        return self._client

    @timing_decorator
    async def ensure_collection_exists(self):
        """
        Ensure the required collection exists with proper configuration.
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [collection.name for collection in collections.collections]

            if self._collection_name not in collection_names:
                # Create collection with specified vector size and distance metric
                self.client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=models.VectorParams(
                        size=self._vector_size,
                        distance=models.Distance[self._distance.upper()]
                    ),
                    # Enable indexing for payload fields we'll be filtering on
                    optimizers_config=models.OptimizersConfigDiff(
                        memmap_threshold=20000,  # Use memory mapping for better performance
                        indexing_threshold=20000  # Start indexing after 20k vectors
                    )
                )
                rag_logger.info(f"Created Qdrant collection: {self._collection_name}")
            else:
                rag_logger.info(f"Qdrant collection already exists: {self._collection_name}")

        except UnexpectedResponse as e:
            if e.status_code == 409:  # Collection already exists
                rag_logger.info(f"Qdrant collection already exists: {self._collection_name}")
            else:
                rag_logger.error(f"Error checking/creating collection: {str(e)}")
                raise
        except Exception as e:
            rag_logger.error(f"Unexpected error in ensure_collection_exists: {str(e)}")
            raise

    @timing_decorator
    async def upsert_points(self, points: List[models.PointStruct]):
        """
        Upsert multiple points into the collection.
        """
        try:
            self.client.upsert(
                collection_name=self._collection_name,
                points=points,
                wait=True  # Wait for operation to complete
            )
            rag_logger.info(f"Upserted {len(points)} points to collection {self._collection_name}")
        except Exception as e:
            rag_logger.error(f"Error upserting points: {str(e)}")
            raise

    @timing_decorator
    async def delete_points(self, point_ids: List[str]):
        """
        Delete points by their IDs.
        """
        try:
            self.client.delete(
                collection_name=self._collection_name,
                points_selector=models.PointIdsList(
                    points=point_ids
                ),
                wait=True
            )
            rag_logger.info(f"Deleted {len(point_ids)} points from collection {self._collection_name}")
        except Exception as e:
            rag_logger.error(f"Error deleting points: {str(e)}")
            raise

    @timing_decorator
    async def batch_insert(self, points: List[models.PointStruct]):
        """
        Batch insert points with retry logic.
        """
        max_retries = 3
        retry_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                await self.upsert_points(points)
                return  # Success, exit the retry loop
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    rag_logger.error(f"Failed to batch insert after {max_retries} attempts: {str(e)}")
                    raise
                else:
                    rag_logger.warning(f"Batch insert attempt {attempt + 1} failed, retrying in {retry_delay}s: {str(e)}")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff

    @timing_decorator
    async def fetch_by_id(self, point_id: str):
        """
        Fetch a single point by ID.
        """
        try:
            records = self.client.retrieve(
                collection_name=self._collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=True
            )
            if records:
                return records[0]
            return None
        except Exception as e:
            rag_logger.error(f"Error fetching point by ID {point_id}: {str(e)}")
            raise

    @timing_decorator
    async def query_by_vector_similarity(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None
    ):
        """
        Query points by vector similarity with optional filtering.
        """
        try:
            # Build Qdrant filter if filters are provided
            qdrant_filter = None
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    if isinstance(value, list):
                        # Handle array values (e.g., document references)
                        filter_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchAny(any=value)
                            )
                        )
                    else:
                        # Handle single values
                        filter_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=value)
                            )
                        )

                if filter_conditions:
                    qdrant_filter = models.Filter(
                        must=filter_conditions
                    )

            # Perform the search - using correct Qdrant API method (query_points with correct parameter)
            search_result = self.client.query_points(
                collection_name=self._collection_name,
                query=query_vector,  # Use 'query' parameter instead of 'query_vector'
                query_filter=qdrant_filter,
                limit=top_k,
                score_threshold=threshold,
                with_payload=True,
                with_vectors=False  # We don't need vectors in response for most use cases
            )

            # Extract the points from the QueryResponse object
            # The query_points method returns a QueryResponse object with a 'results' attribute
            if hasattr(search_result, 'results'):
                search_results = search_result.results
            elif hasattr(search_result, 'points'):
                search_results = search_result.points
            else:
                search_results = search_result

            # Ensure search_results is a list for consistency
            if not isinstance(search_results, list):
                if hasattr(search_results, '__iter__') and not isinstance(search_results, (str, bytes)):
                    search_results = list(search_results)
                else:
                    search_results = [] if search_results is None else [search_results]

            result_count = len(search_results)
            rag_logger.info(f"Found {result_count} similar vectors for query")
            return search_results

        except Exception as e:
            rag_logger.error(f"Error in vector similarity search: {str(e)}")
            raise