"""
Google Gemini API client wrapper
"""
import asyncio
import os
import time
from typing import List, Dict, Any, Optional, Union
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Correct import based on documentation
from google import genai

logger = logging.getLogger(__name__)


class GeminiClient:
    """Google Gemini API client wrapper"""

    def __init__(self, api_key: Optional[str] = None):
        # Get API key from environment if not provided
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        # Initialize the client as per documentation
        self.client = genai.Client()

        # Set the embedding model
        self.model_name = os.getenv("EMBED_MODEL_NAME", "gemini-embedding-001")  # Correct model name from config

        # Set the output dimension - default to 1536 as specified in requirements
        self.output_dimensionality = int(os.getenv("EMBEDDING_DIMENSION", "1536"))

        # Track API usage
        self.request_count = 0
        self.last_request_time = 0

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
    async def embed_content(self,
                           content: Union[str, List[str]],
                           task_type: Optional[str] = "SEMANTIC_SIMILARITY",
                           output_dimensionality: Optional[int] = None) -> List[List[float]]:
        """
        Generate embeddings for content using Google Gemini API

        Args:
            content: Single string or list of strings to embed
            task_type: Type of task (SEMANTIC_SIMILARITY, RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, etc.)
            output_dimensionality: Desired output dimension (optional, defaults to configured value)

        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        try:
            # Track request for rate limiting
            self.request_count += 1
            current_time = time.time()

            # Use instance default dimension if not provided
            if output_dimensionality is None:
                output_dimensionality = self.output_dimensionality

            # Prepare request parameters as per documentation
            request_params = {
                'model': self.model_name,
                'contents': content  # Note: documentation uses 'contents', not 'content'
            }

            # Add config if task_type or output_dimensionality is specified
            if task_type or output_dimensionality:
                # Import types as per documentation
                from google.genai import types
                config_params = {}
                if task_type:
                    config_params['task_type'] = task_type
                if output_dimensionality:
                    config_params['output_dimensionality'] = output_dimensionality
                request_params['config'] = types.EmbedContentConfig(**config_params)

            # Generate embeddings using the Google Generative AI library as per documentation
            # Note: client.models.embed_content is the correct method
            loop = asyncio.get_event_loop()

            def sync_embed():
                result = self.client.models.embed_content(**request_params)
                return result

            result = await loop.run_in_executor(None, sync_embed)

            # Extract the embedding values from the result as per documentation
            embeddings = []

            # According to documentation, result.embeddings contains the embeddings
            for embedding_obj in result.embeddings:
                # Extract the values from the embedding object
                if hasattr(embedding_obj, 'values'):
                    embeddings.append(embedding_obj.values)
                else:
                    # Fallback: if it's already a list or similar structure
                    embeddings.append(list(embedding_obj))

            # Update last request time
            self.last_request_time = current_time

            return embeddings

        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            raise

    async def batch_embed_content(self,
                                 content_list: List[str],
                                 batch_size: int = 5,  # Reduced batch size for safety
                                 task_type: Optional[str] = "SEMANTIC_SIMILARITY",
                                 output_dimensionality: Optional[int] = None) -> List[List[float]]:
        """
        Generate embeddings for multiple content items in batches

        Args:
            content_list: List of strings to embed
            batch_size: Number of items to process in each batch
            task_type: Type of task (SEMANTIC_SIMILARITY, RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, etc.)
            output_dimensionality: Desired output dimension (optional)

        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        all_embeddings = []

        # Use instance default dimension if not provided
        if output_dimensionality is None:
            output_dimensionality = self.output_dimensionality

        # Process in batches to respect API limits
        for i in range(0, len(content_list), batch_size):
            batch = content_list[i:i + batch_size]

            try:
                # Process each item in the batch
                for item in batch:
                    item_embeddings = await self.embed_content(
                        item,
                        task_type=task_type,
                        output_dimensionality=output_dimensionality
                    )
                    all_embeddings.extend(item_embeddings)

                # Small delay between batches to respect rate limits
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Batch embedding failed for batch starting at index {i}: {str(e)}")
                # Add None placeholders for failed embeddings to maintain order
                all_embeddings.extend([None] * len(batch))

        # Remove None values and return only successful embeddings
        return [emb for emb in all_embeddings if emb is not None]

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        return {
            "request_count": self.request_count,
            "last_request_time": self.last_request_time,
            "model_used": self.model_name,
            "configured_output_dimensionality": self.output_dimensionality
        }

    async def validate_api_key(self) -> bool:
        """Validate that the API key is working by making a test call"""
        try:
            # Try embedding a simple test string
            test_embedding = await self.embed_content("test")
            return len(test_embedding) > 0 and len(test_embedding[0]) > 0 if test_embedding else False
        except Exception as e:
            logger.error(f"API key validation failed: {str(e)}")
            return False


class EmbeddingProcessor:
    """Class for Google Gemini API integration"""

    def __init__(self):
        self.client = None
        self.batch_size = int(os.getenv("BATCH_SIZE", "5"))  # Reduced for safety
        # Use the same embedding dimension as configured for Qdrant
        self.output_dimensionality = int(os.getenv("EMBEDDING_DIMENSION", "1536"))

    async def initialize(self):
        """Initialize the Gemini client"""
        self.client = GeminiClient()
        # Validate the API key
        is_valid = await self.client.validate_api_key()
        if not is_valid:
            raise ValueError("Invalid Gemini API key")

    async def generate_embeddings(self, chunks: List['Chunk']) -> List[List[float]]:
        """Generate embeddings for the provided chunks"""
        if not self.client:
            await self.initialize()

        # Extract content from chunks
        content_list = [chunk.content for chunk in chunks]

        # Generate embeddings in batches with the configured output dimension
        embeddings = await self.client.batch_embed_content(
            content_list,
            batch_size=self.batch_size,
            output_dimensionality=self.output_dimensionality
        )

        # Verify that we got embeddings for all chunks
        if len(embeddings) != len(chunks):
            logger.warning(f"Mismatch in embedding count: expected {len(chunks)}, got {len(embeddings)}")

        return embeddings

    def get_client_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        stats = {}
        if self.client:
            stats.update(self.client.get_usage_stats())
        # Add embedding processor stats
        stats['configured_output_dimensionality'] = self.output_dimensionality
        return stats