from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from qdrant_client import models


class QdrantPayload(BaseModel):
    """
    Schema for Qdrant payload structure.
    """
    text: str
    document_reference: str
    metadata: Dict[str, Any] = {}
    page_reference: Optional[int] = None
    section_title: Optional[str] = None
    chunk_index: Optional[int] = None
    source: Optional[str] = None
    module: Optional[str] = None
    processing_version: Optional[str] = None
    created_at: Optional[str] = None

    def to_payload_dict(self) -> Dict[str, Any]:
        """
        Convert the payload model to a dictionary for Qdrant.
        """
        return self.dict(exclude_none=True)

    @classmethod
    def from_payload_dict(cls, payload: Dict[str, Any]) -> 'QdrantPayload':
        """
        Create a QdrantPayload from a payload dictionary.
        """
        return cls(**payload)


class QdrantPoint(BaseModel):
    """
    Schema for a complete Qdrant point (ID, vector, payload).
    """
    id: str
    vector: List[float]
    payload: QdrantPayload

    def to_point_struct(self) -> models.PointStruct:
        """
        Convert to Qdrant PointStruct for insertion.
        """
        return models.PointStruct(
            id=self.id,
            vector=self.vector,
            payload=self.payload.to_payload_dict()
        )

    @classmethod
    def from_point_struct(cls, point: models.PointStruct) -> 'QdrantPoint':
        """
        Create a QdrantPoint from a PointStruct.
        """
        payload = QdrantPayload.from_payload_dict(point.payload)
        return cls(
            id=point.id,
            vector=point.vector,
            payload=payload
        )


class QdrantSearchFilter(BaseModel):
    """
    Schema for Qdrant search filters.
    """
    document_reference: Optional[str] = None
    page_reference: Optional[int] = None
    section_title: Optional[str] = None
    source: Optional[str] = None
    module: Optional[str] = None
    metadata_filters: Optional[Dict[str, Any]] = None

    def to_qdrant_filter(self) -> Optional[models.Filter]:
        """
        Convert to Qdrant Filter object.
        """
        if not any([
            self.document_reference, self.page_reference, self.section_title,
            self.source, self.module, self.metadata_filters
        ]):
            return None

        filter_conditions = []

        if self.document_reference:
            filter_conditions.append(
                models.FieldCondition(
                    key="document_reference",
                    match=models.MatchValue(value=self.document_reference)
                )
            )

        if self.page_reference is not None:
            filter_conditions.append(
                models.FieldCondition(
                    key="page_reference",
                    match=models.MatchValue(value=self.page_reference)
                )
            )

        if self.section_title:
            filter_conditions.append(
                models.FieldCondition(
                    key="section_title",
                    match=models.MatchValue(value=self.section_title)
                )
            )

        if self.source:
            filter_conditions.append(
                models.FieldCondition(
                    key="source",
                    match=models.MatchValue(value=self.source)
                )
            )

        if self.module:
            filter_conditions.append(
                models.FieldCondition(
                    key="module",
                    match=models.MatchValue(value=self.module)
                )
            )

        if self.metadata_filters:
            for key, value in self.metadata_filters.items():
                filter_conditions.append(
                    models.FieldCondition(
                        key=f"metadata.{key}",
                        match=models.MatchValue(value=value)
                    )
                )

        if filter_conditions:
            return models.Filter(must=filter_conditions)

        return None


class QdrantCollectionConfig(BaseModel):
    """
    Schema for Qdrant collection configuration.
    """
    collection_name: str
    vector_size: int = 1024  # Default to 1024 for Cohere embeddings
    distance: str = "Cosine"
    shard_number: Optional[int] = None
    replication_factor: Optional[int] = None
    write_consistency_factor: Optional[int] = None
    on_disk_payload: bool = False

    def to_collection_config(self) -> models.CreateCollection:
        """
        Convert to Qdrant CreateCollection model.
        """
        return models.CreateCollection(
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance[self.distance.upper()]
            ),
            shard_number=self.shard_number,
            replication_factor=self.replication_factor,
            write_consistency_factor=self.write_consistency_factor,
            on_disk_payload=self.on_disk_payload
        )