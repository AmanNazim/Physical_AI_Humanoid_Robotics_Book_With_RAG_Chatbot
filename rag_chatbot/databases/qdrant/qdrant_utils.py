from typing import List, Dict, Any, Optional
from qdrant_client import models
from rag_core.utils.logger import rag_logger
from rag_core.utils.timing import timing_decorator
from .qdrant_schema import QdrantPayload, QdrantSearchFilter


class QdrantUtils:
    """
    Utility functions for working with Qdrant data structures.
    """

    @staticmethod
    @timing_decorator
    def build_payload_from_chunk(
        text: str,
        document_reference: str,
        chunk_id: str,
        page_reference: Optional[int] = None,
        section_title: Optional[str] = None,
        source: Optional[str] = None,
        module: Optional[str] = None,
        processing_version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> QdrantPayload:
        """
        Build a Qdrant payload from chunk information.
        """
        payload = QdrantPayload(
            text=text,
            document_reference=document_reference,
            page_reference=page_reference,
            section_title=section_title,
            source=source,
            module=module,
            processing_version=processing_version,
            created_at="timestamp_placeholder",  # Will be set by system
            metadata=metadata or {}
        )
        return payload

    @staticmethod
    @timing_decorator
    def convert_raw_query_to_qdrant_query(
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Convert raw query parameters to Qdrant search parameters.
        """
        search_params = {
            "query_text": query,
            "top_k": top_k,
            "threshold": threshold
        }

        if filters:
            search_params["filters"] = filters

        return search_params

    @staticmethod
    @timing_decorator
    def map_qdrant_results_to_internal_schema(
        qdrant_results: List[models.ScoredPoint]
    ) -> List[Dict[str, Any]]:
        """
        Map Qdrant search results to internal retrieval schema.
        """
        internal_results = []
        for result in qdrant_results:
            payload = result.payload or {}
            internal_result = {
                "chunk_id": result.id,
                "text": payload.get("content", ""),  # Changed from "text" to "content"
                "document_reference": payload.get("document_reference", ""),
                "score": result.score or 0.0,
                "page_reference": payload.get("page_reference"),
                "section_title": payload.get("section_title"),
                "metadata": payload.get("metadata", {}),
                "source": payload.get("source"),
                "module": payload.get("module")
            }
            internal_results.append(internal_result)

        rag_logger.info(f"Mapped {len(internal_results)} Qdrant results to internal schema")
        return internal_results

    @staticmethod
    @timing_decorator
    def create_qdrant_filter_from_dict(
        filters: Optional[Dict[str, Any]] = None
    ) -> Optional[models.Filter]:
        """
        Create a Qdrant filter from a dictionary of filter parameters.
        """
        if not filters:
            return None

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
            return models.Filter(must=filter_conditions)

        return None

    @staticmethod
    @timing_decorator
    def validate_embedding_vector(
        vector: List[float],
        expected_dimension: int = 1024
    ) -> bool:
        """
        Validate that an embedding vector has the correct dimensionality.
        """
        if not isinstance(vector, list):
            rag_logger.error("Embedding vector must be a list")
            return False

        if len(vector) != expected_dimension:
            rag_logger.error(f"Embedding vector dimension mismatch: expected {expected_dimension}, got {len(vector)}")
            return False

        # Check that all values are floats
        for i, val in enumerate(vector):
            if not isinstance(val, (int, float)):
                rag_logger.error(f"Invalid value at index {i}: {val} is not a number")
                return False

        rag_logger.info(f"Validated embedding vector with {len(vector)} dimensions")
        return True

    @staticmethod
    @timing_decorator
    def extract_document_references_from_payloads(
        payloads: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Extract document references from a list of payloads.
        """
        document_refs = []
        for payload in payloads:
            doc_ref = payload.get("document_reference")
            if doc_ref:
                document_refs.append(doc_ref)
        return list(set(document_refs))  # Return unique document references

    @staticmethod
    @timing_decorator
    def build_qdrant_point(
        vector: List[float],
        chunk_id: str,
        payload: QdrantPayload
    ) -> models.PointStruct:
        """
        Build a Qdrant PointStruct from vector, ID, and payload.
        """
        if not QdrantUtils.validate_embedding_vector(vector):
            raise ValueError(f"Invalid embedding vector for chunk {chunk_id}")

        return models.PointStruct(
            id=chunk_id,
            vector=vector,
            payload=payload.to_payload_dict()
        )

    @staticmethod
    @timing_decorator
    def build_batch_points(
        vectors: List[List[float]],
        chunk_ids: List[str],
        payloads: List[QdrantPayload]
    ) -> List[models.PointStruct]:
        """
        Build multiple Qdrant PointStruct objects for batch operations.
        """
        if not (len(vectors) == len(chunk_ids) == len(payloads)):
            raise ValueError("Vectors, chunk_ids, and payloads must have the same length")

        points = []
        for i, (vector, chunk_id, payload) in enumerate(zip(vectors, chunk_ids, payloads)):
            try:
                point = QdrantUtils.build_qdrant_point(vector, chunk_id, payload)
                points.append(point)
            except Exception as e:
                rag_logger.error(f"Error building point {i} (ID: {chunk_id}): {str(e)}")
                raise

        rag_logger.info(f"Built {len(points)} points for batch operation")
        return points

    @staticmethod
    @timing_decorator
    def extract_metadata_fields(
        payloads: List[Dict[str, Any]],
        field_name: str
    ) -> List[Any]:
        """
        Extract a specific field from metadata across multiple payloads.
        """
        values = []
        for payload in payloads:
            metadata = payload.get("metadata", {})
            value = metadata.get(field_name)
            if value is not None:
                values.append(value)
        return values