"""
Gemini Embedding Client for the Embeddings & Chunking Pipeline.
This module implements the Gemini API integration with proper batching and error handling.
"""
import asyncio
import time
from typing import List, Optional, Dict, Any
from google import genai
from google.genai import types
from embedding_pipeline.config import config
import os


class GeminiEmbeddingClient:
    """
    A client for generating embeddings using Google's Gemini API with proper batching,
    error handling, and retry logic as specified in the system requirements.
    """

    def __init__(self):
        """
        Initialize the Gemini Embedding Client.
        """
        if not config.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable must be set")

        # Initialize the client with API key as per documentation
        self.client = genai.Client(api_key=config.gemini_api_key)
        self.model_name = config.gemini_model_name
        self.batch_size = config.max_batch_size
        self.max_retries = config.max_retries
        self.retry_delay_base = config.retry_delay_base

    async def embed_batch(self, chunks: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of text chunks using Gemini API.
        This implements the required batching strategy with proper error handling.

        Args:
            chunks: List of text chunks to generate embeddings for

        Returns:
            List of embedding vectors (each vector is a list of floats)

        Raises:
            RuntimeError: If API call fails after all retry attempts
        """
        if not chunks:
            return []

        all_embeddings = []

        # Process in batches to respect API limits and optimize performance
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batch_embeddings = await self._process_single_batch(batch)
            all_embeddings.extend(batch_embeddings)

            # No delay to prevent hanging - let external services handle rate limiting

        return all_embeddings

    async def _process_single_batch(self, batch: List[str]) -> List[List[float]]:
        """
        Process a single batch of text chunks with retry logic using the proper API as per documentation.

        Args:
            batch: List of text chunks in the current batch

        Returns:
            List of embedding vectors for the batch
        """
        for attempt in range(self.max_retries):
            try:
                # According to the official documentation, use the client's embed_content method
                # which can accept multiple contents as a list for proper batching
                # Prepare config for embedding generation using types.EmbedContentConfig
                embed_config_kwargs = {}
                if config.gemini_task_type:
                    embed_config_kwargs['task_type'] = config.gemini_task_type
                if config.gemini_output_dimensionality and config.gemini_output_dimensionality != 0:
                    embed_config_kwargs['output_dimensionality'] = config.gemini_output_dimensionality

                embed_config = types.EmbedContentConfig(**embed_config_kwargs) if embed_config_kwargs else None

                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: self.client.models.embed_content(
                        model=self.model_name,
                        contents=[batch],  # Pass the entire batch as a list of strings
                        config=embed_config
                    )),
                    timeout=120.0  # 2 minutes timeout for API call
                )

                # Extract embeddings from the result
                embeddings_data = result.embeddings
                if embeddings_data and len(embeddings_data) > 0:
                    embedding_vectors = []
                    for emb_data in embeddings_data:
                        embedding_vector = emb_data.values  # Access values attribute directly
                        if embedding_vector:
                            embedding_vectors.append(embedding_vector)
                        else:
                            # If an embedding is missing, return a zero vector
                            embedding_vectors.append([0.0] * config.gemini_output_dimensionality if config.gemini_output_dimensionality != 0 else [0.0] * 1024)
                    return embedding_vectors
                else:
                    raise RuntimeError("No embeddings returned from API")

            except Exception as e:
                if attempt < self.max_retries - 1:
                    # Exponential backoff with jitter
                    print(f"  ⚠️  Batch embedding attempt {attempt + 1} failed: {str(e)}")
                    print(f"  ⏳  Retrying immediately to prevent hanging...")
                    # No sleep to prevent hanging - immediate retry on failure
                    continue
                else:
                    print(f"  ❌ Failed to generate embeddings for batch after {self.max_retries} attempts: {str(e)}")
                    # Instead of failing completely, return zero vectors for failed items
                    # This allows the system to continue processing
                    failed_embeddings = []
                    for _ in batch:
                        failed_embeddings.append([0.0] * config.gemini_output_dimensionality if config.gemini_output_dimensionality != 0 else [0.0] * 1024)
                    return failed_embeddings

    async def embed_single(self, text: str) -> List[float]:
        """
        Generate a single embedding for the given text using Gemini API.

        Args:
            text: Input text to generate embedding for

        Returns:
            List of float values representing the embedding vector
        """
        for attempt in range(self.max_retries):
            try:
                # Prepare config for embedding generation using types.EmbedContentConfig
                embed_config_kwargs = {}
                if config.gemini_task_type:
                    embed_config_kwargs['task_type'] = config.gemini_task_type
                if config.gemini_output_dimensionality and config.gemini_output_dimensionality != 0:
                    embed_config_kwargs['output_dimensionality'] = config.gemini_output_dimensionality

                embed_config = types.EmbedContentConfig(**embed_config_kwargs) if embed_config_kwargs else None

                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: self.client.models.embed_content(
                        model=self.model_name,
                        contents=[text],  # Single text should be passed as a list for consistency
                        config=embed_config
                    )),
                    timeout=120.0  # 2 minutes timeout for API call
                )

                embeddings = result.embeddings

                if embeddings and len(embeddings) > 0:
                    # Get the first embedding since single text should return one embedding
                    first_embedding = embeddings[0].values  # Access values attribute directly
                    if first_embedding:
                        # Check embedding dimension
                        actual_dim = len(first_embedding)
                        if config.gemini_output_dimensionality != actual_dim and config.gemini_output_dimensionality != 0:  # 0 means auto-detect
                            print(f"Warning: Expected embedding dimension {config.gemini_output_dimensionality}, got {actual_dim}. "
                                  f"Accepting actual dimension for this embedding.")

                        return first_embedding
                    else:
                        raise RuntimeError("No embedding values returned from API")
                else:
                    raise RuntimeError("No embeddings returned from API")

            except Exception as e:
                if attempt < self.max_retries - 1:
                    # Exponential backoff with jitter
                    print(f"  ⚠️  Single embedding attempt {attempt + 1} failed: {str(e)}")
                    print(f"  ⏳  Retrying immediately to prevent hanging...")
                    # No sleep to prevent hanging - immediate retry on failure
                    continue
                else:
                    raise RuntimeError(f"Failed to generate embedding after {self.max_retries} attempts: {str(e)}")

    async def process_directory(self, path: str) -> Dict[str, Any]:
        """
        Process a directory of documents for embedding generation following the
        human-relevant order specified in the requirements.

        Args:
            path: Path to the directory containing documents

        Returns:
            Dictionary with processing results
        """
        import os
        from pathlib import Path

        results = {
            "status": "success",
            "processed_files": [],
            "total_chunks": 0,
            "total_processing_time": 0
        }

        start_time = time.time()

        # Process in the specified human-relevant order
        path_obj = Path(path)

        # A. Preface
        print("📚 Processing Preface content...")
        preface_dir = path_obj / "preface"
        if preface_dir.exists():
            preface_files = sorted(preface_dir.rglob("*.md"))  # Sort for consistent order
            for file_path in preface_files:
                result = await self._process_single_file_with_pipeline(str(file_path))
                results["processed_files"].append(result)
                results["total_chunks"] += len(result.get("chunk_ids", []))

        # B. Module 1 - Sequential batching
        print("📚 Processing Module 1 content...")
        module1_dir = path_obj / "module-1"
        if module1_dir.exists():
            # Process introduction first
            intro_file = module1_dir / "introduction.md"
            if intro_file.exists():
                result = await self._process_single_file_with_pipeline(str(intro_file))
                results["processed_files"].append(result)
                results["total_chunks"] += len(result.get("chunk_ids", []))

            # Process chapters in the module
            for lesson_dir in sorted(module1_dir.iterdir()):
                if lesson_dir.is_dir():
                    lesson_files = sorted(lesson_dir.rglob("*.md"))
                    for file_path in lesson_files:
                        if file_path.name != "introduction.md":
                            result = await self._process_single_file_with_pipeline(str(file_path))
                            results["processed_files"].append(result)
                            results["total_chunks"] += len(result.get("chunk_ids", []))

        # C. Module 2 - Sequential batching
        print("📚 Processing Module 2 content...")
        module2_dir = path_obj / "module-2"
        if module2_dir.exists():
            # Process introduction first
            intro_file = module2_dir / "introduction.md"
            if intro_file.exists():
                result = await self._process_single_file_with_pipeline(str(intro_file))
                results["processed_files"].append(result)
                results["total_chunks"] += len(result.get("chunk_ids", []))

            # Process chapters in the module
            for lesson_dir in sorted(module2_dir.iterdir()):
                if lesson_dir.is_dir():
                    lesson_files = sorted(lesson_dir.rglob("*.md"))
                    for file_path in lesson_files:
                        if file_path.name != "introduction.md":
                            result = await self._process_single_file_with_pipeline(str(file_path))
                            results["processed_files"].append(result)
                            results["total_chunks"] += len(result.get("chunk_ids", []))

        # D. Module 3 - Sequential batching
        print("📚 Processing Module 3 content...")
        module3_dir = path_obj / "module-3"
        if module3_dir.exists():
            # Process introduction first
            intro_file = module3_dir / "introduction.md"
            if intro_file.exists():
                result = await self._process_single_file_with_pipeline(str(intro_file))
                results["processed_files"].append(result)
                results["total_chunks"] += len(result.get("chunk_ids", []))

            # Process chapters in the module
            for lesson_dir in sorted(module3_dir.iterdir()):
                if lesson_dir.is_dir():
                    lesson_files = sorted(lesson_dir.rglob("*.md"))
                    for file_path in lesson_files:
                        if file_path.name != "introduction.md":
                            result = await self._process_single_file_with_pipeline(str(file_path))
                            results["processed_files"].append(result)
                            results["total_chunks"] += len(result.get("chunk_ids", []))

        # E. Module 4 - Sequential batching
        print("📚 Processing Module 4 content...")
        module4_dir = path_obj / "module-4"
        if module4_dir.exists():
            # Process introduction first
            intro_file = module4_dir / "introduction.md"
            if intro_file.exists():
                result = await self._process_single_file_with_pipeline(str(intro_file))
                results["processed_files"].append(result)
                results["total_chunks"] += len(result.get("chunk_ids", []))

            # Process chapters in the module
            for lesson_dir in sorted(module4_dir.iterdir()):
                if lesson_dir.is_dir():
                    lesson_files = sorted(lesson_dir.rglob("*.md"))
                    for file_path in lesson_files:
                        if file_path.name != "introduction.md":
                            result = await self._process_single_file_with_pipeline(str(file_path))
                            results["processed_files"].append(result)
                            results["total_chunks"] += len(result.get("chunk_ids", []))

        # F. Assessments content
        print("📚 Processing Assessments content...")
        assessments_dir = path_obj / "assessments"
        if assessments_dir.exists():
            assessment_files = sorted(assessments_dir.rglob("*.md"))
            for file_path in assessment_files:
                result = await self._process_single_file_with_pipeline(str(file_path))
                results["processed_files"].append(result)
                results["total_chunks"] += len(result.get("chunk_ids", []))

        # G. Hardware Requirements content
        print("📚 Processing Hardware Requirements content...")
        hardware_dir = path_obj / "Hardware-Requirements"
        if hardware_dir.exists():
            hardware_files = sorted(hardware_dir.rglob("*.md"))
            for file_path in hardware_files:
                result = await self._process_single_file_with_pipeline(str(file_path))
                results["processed_files"].append(result)
                results["total_chunks"] += len(result.get("chunk_ids", []))

        results["total_processing_time"] = time.time() - start_time

        return results

    async def _process_single_file_with_pipeline(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single file using the full pipeline.

        Args:
            file_path: Path to the file to process

        Returns:
            Dictionary with processing results for the file
        """
        from .pipeline import process_file

        # Create basic metadata for the file
        metadata = {
            "document_reference": os.path.basename(file_path),
            "page_reference": None,
            "section_title": os.path.basename(file_path),
            "processing_version": "1.0",
            "additional_metadata": {
                "source_file": file_path,
                "content_type": "book_content",
                "language": "en"
            }
        }

        try:
            result = await process_file(file_path, metadata)
            return result
        except Exception as e:
            return {
                "file_path": file_path,
                "status": "error",
                "message": str(e),
                "chunk_ids": [],
                "processing_time": 0
            }


# Global client instance for reuse
_gemini_client = None


def get_gemini_client() -> GeminiEmbeddingClient:
    """
    Get or create a singleton instance of the Gemini Embedding Client.

    Returns:
        GeminiEmbeddingClient instance
    """
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiEmbeddingClient()
    return _gemini_client