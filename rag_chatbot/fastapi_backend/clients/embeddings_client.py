from typing import List, Dict, Any, Optional
from ..embedding_pipeline.gemini_client import EmbeddingProcessor
from ..utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingsClient:
    """
    Embeddings client wrapper for the FastAPI backend.
    This client provides a clean interface to interact with the Embeddings subsystem.
    """

    def __init__(self):
        self.processor = EmbeddingProcessor()

    async def initialize(self):
        """
        Initialize the embeddings processor.
        """
        await self.processor.initialize()

    async def generate_embeddings(
        self,
        texts: List[str],
        task_type: str = "SEMANTIC_SIMILARITY",
        output_dimensionality: Optional[int] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for the provided texts.

        Args:
            texts: List of texts to generate embeddings for
            task_type: Type of embedding task (SEMANTIC_SIMILARITY, RETRIEVAL_DOCUMENT, etc.)
            output_dimensionality: Desired output dimension (defaults to configured value)

        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        try:
            # Create temporary chunk objects to work with the existing processor
            class TempChunk:
                def __init__(self, content: str, idx: int):
                    self.content = content
                    self.id = f"temp_chunk_{idx}"

            # Create chunk objects from the texts
            chunks = [TempChunk(text, i) for i, text in enumerate(texts)]

            # Generate embeddings using the existing processor
            embeddings = await self.processor.generate_embeddings(chunks)

            logger.info(f"Generated {len(embeddings)} embeddings for {len(texts)} texts")
            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    async def embed_single_text(
        self,
        text: str,
        task_type: str = "SEMANTIC_SIMILARITY",
        output_dimensionality: Optional[int] = None
    ) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to generate embedding for
            task_type: Type of embedding task
            output_dimensionality: Desired output dimension

        Returns:
            Embedding vector (list of floats)
        """
        embeddings = await self.generate_embeddings([text], task_type, output_dimensionality)
        if embeddings and len(embeddings) > 0:
            return embeddings[0]
        else:
            raise ValueError("Failed to generate embedding for the provided text")

    async def get_client_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the embeddings client.

        Returns:
            Dictionary containing client statistics
        """
        return self.processor.get_client_stats()