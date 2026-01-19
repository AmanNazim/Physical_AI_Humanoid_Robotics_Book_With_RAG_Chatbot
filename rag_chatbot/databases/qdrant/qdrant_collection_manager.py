import uuid
from typing import List, Optional, Dict, Any
from qdrant_client import models
from rag_core.utils.logger import rag_logger
from rag_core.utils.timing import timing_decorator
from .qdrant_client import QdrantClientWrapper
from shared.schemas.retrieval import RetrievalResult


class QdrantCollectionManager:
    """
    Manages Qdrant collection operations including CRUD and query operations.
    """

    def __init__(self, qdrant_client: QdrantClientWrapper):
        self.qdrant_client = qdrant_client

    @timing_decorator
    async def create_collection(self, collection_name: str, vector_size: int, distance: str = "Cosine"):
        """
        Create a Qdrant collection with specified parameters.
        """
        try:
            await self.qdrant_client.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance[distance.upper()]
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    memmap_threshold=20000,
                    indexing_threshold=20000
                )
            )
            rag_logger.info(f"Created Qdrant collection: {collection_name}")
        except Exception as e:
            rag_logger.error(f"Error creating collection {collection_name}: {str(e)}")
            raise

    @timing_decorator
    async def validate_collection(self, collection_name: str) -> bool:
        """
        Validate that a collection exists and is accessible.
        """
        try:
            collections = await self.qdrant_client.client.get_collections()
            collection_names = [collection.name for collection in collections.collections]
            exists = collection_name in collection_names

            if exists:
                rag_logger.info(f"Collection {collection_name} validated successfully")
            else:
                rag_logger.warning(f"Collection {collection_name} does not exist")

            return exists
        except Exception as e:
            rag_logger.error(f"Error validating collection {collection_name}: {str(e)}")
            return False

    @timing_decorator
    async def upsert_points(self, points: List[models.PointStruct]) -> bool:
        """
        Upsert points into the collection.
        """
        try:
            await self.qdrant_client.upsert_points(points)
            rag_logger.info(f"Successfully upserted {len(points)} points")
            return True
        except Exception as e:
            rag_logger.error(f"Error upserting points: {str(e)}")
            return False

    @timing_decorator
    async def delete_points(self, point_ids: List[str]) -> bool:
        """
        Delete points by their IDs.
        """
        try:
            await self.qdrant_client.delete_points(point_ids)
            rag_logger.info(f"Successfully deleted {len(point_ids)} points")
            return True
        except Exception as e:
            rag_logger.error(f"Error deleting points: {str(e)}")
            return False

    @timing_decorator
    async def batch_insert(self, points: List[models.PointStruct]) -> bool:
        """
        Batch insert points with retry logic.
        """
        try:
            await self.qdrant_client.batch_insert(points)
            rag_logger.info(f"Successfully batch inserted {len(points)} points")
            return True
        except Exception as e:
            rag_logger.error(f"Error in batch insert: {str(e)}")
            return False

    @timing_decorator
    async def fetch_by_id(self, point_id: str) -> Optional[models.Record]:
        """
        Fetch a single point by ID.
        """
        try:
            result = await self.qdrant_client.fetch_by_id(point_id)
            if result:
                rag_logger.info(f"Fetched point by ID: {point_id}")
                return result
            else:
                rag_logger.info(f"No point found with ID: {point_id}")
                return None
        except Exception as e:
            rag_logger.error(f"Error fetching point by ID {point_id}: {str(e)}")
            return None

    @timing_decorator
    async def query_by_vector_similarity(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None
    ) -> List[RetrievalResult]:
        """
        Query points by vector similarity and return structured results.
        """
        try:
            search_results = await self.qdrant_client.query_by_vector_similarity(
                query_vector=query_vector,
                top_k=top_k,
                filters=filters,
                threshold=threshold
            )

            # Convert Qdrant results to our internal format
            retrieval_results = []
            for result in search_results:
                payload = result.payload or {}
                retrieval_result = RetrievalResult(
                    chunk_id=result.id,
                    text=payload.get("content", ""),  # Changed from "text" to "content"
                    document_reference=payload.get("document_reference", ""),
                    score=result.score or 0.0,
                    page_reference=payload.get("page_reference"),
                    section_title=payload.get("section_title"),
                    metadata=payload.get("metadata", {})
                )
                retrieval_results.append(retrieval_result)

            rag_logger.info(f"Retrieved {len(retrieval_results)} results from vector similarity search")
            return retrieval_results

        except Exception as e:
            rag_logger.error(f"Error in vector similarity search: {str(e)}")
            return []

    @timing_decorator
    async def create_point_struct(
        self,
        vector: List[float],
        chunk_id: str,
        text: str,
        document_reference: str,
        page_reference: Optional[int] = None,
        section_title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> models.PointStruct:
        """
        Create a PointStruct with proper payload structure.
        """
        payload = {
            "text": text,
            "document_reference": document_reference,
            "metadata": metadata or {}
        }

        if page_reference is not None:
            payload["page_reference"] = page_reference

        if section_title is not None:
            payload["section_title"] = section_title

        return models.PointStruct(
            id=chunk_id,
            vector=vector,
            payload=payload
        )

    @timing_decorator
    async def batch_create_points(
        self,
        vectors: List[List[float]],
        texts: List[str],
        document_references: List[str],
        chunk_ids: Optional[List[str]] = None,
        page_references: Optional[List[Optional[int]]] = None,
        section_titles: Optional[List[Optional[str]]] = None,
        metadatas: Optional[List[Optional[Dict[str, Any]]]] = None
    ) -> List[models.PointStruct]:
        """
        Create multiple PointStruct objects efficiently.
        """
        if not all(len(lst) == len(vectors) for lst in [texts, document_references]):
            raise ValueError("All input lists must have the same length as vectors")

        points = []
        for i, vector in enumerate(vectors):
            chunk_id = chunk_ids[i] if chunk_ids else str(uuid.uuid4())
            page_ref = page_references[i] if page_references else None
            section_title = section_titles[i] if section_titles else None
            metadata = metadatas[i] if metadatas else None

            point = await self.create_point_struct(
                vector=vector,
                chunk_id=chunk_id,
                text=texts[i],
                document_reference=document_references[i],
                page_reference=page_ref,
                section_title=section_title,
                metadata=metadata
            )
            points.append(point)

        return points