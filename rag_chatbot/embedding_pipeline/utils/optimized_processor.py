"""
Optimized processor for fastest embedding generation and storage
"""
import asyncio
import time
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProcessStats:
    """Statistics for the optimized processing"""
    total_chunks: int = 0
    processed_chunks: int = 0
    total_tokens: int = 0
    processing_time_ms: float = 0.0
    embedding_time_ms: float = 0.0
    storage_time_ms: float = 0.0


class OptimizedProcessor:
    """Class for optimized processing for fastest embedding generation and storage"""

    def __init__(self):
        self.stats = ProcessStats()

    async def process_single_url_optimized(self, url: str, text_content: str) -> bool:
        """
        Process a single URL with optimized code for fastest embedding generation and storage
        """
        start_time = time.time()

        try:
            # Update stats
            self.stats.total_chunks += 1
            self.stats.processed_chunks += 1

            # Process the content efficiently
            from ..chunking_engine import ChunkProcessor
            chunk_processor = ChunkProcessor()

            # Chunk the content efficiently
            chunks = chunk_processor.chunk_text(text_content, document_reference=url)

            # Update token count
            for chunk in chunks:
                self.stats.total_tokens += chunk.token_count

            # Generate embeddings efficiently
            from ..gemini_client import EmbeddingProcessor
            embedding_processor = EmbeddingProcessor()

            embedding_start = time.time()
            embeddings = await embedding_processor.generate_embeddings(chunks)
            self.stats.embedding_time_ms += (time.time() - embedding_start) * 1000

            # Store efficiently
            from ..database import DatabaseManager
            db_processor = DatabaseManager()

            storage_start = time.time()
            success = await db_processor.store_embeddings_with_metadata(chunks, embeddings)
            self.stats.storage_time_ms += (time.time() - storage_start) * 1000

            self.stats.processing_time_ms = (time.time() - start_time) * 1000

            return success
        except Exception as e:
            logger.error(f"Failed to process URL {url}: {str(e)}")
            return False

    async def process_multiple_urls_optimized(self, urls: List[str]) -> Dict[str, Any]:
        """
        Process multiple URLs with optimized code for fastest embedding generation and storage
        """
        start_time = time.time()
        results = []

        for url in urls:
            # Fetch content
            try:
                from ..url_crawler import URLCrawler
                url_crawler = URLCrawler()
                content = await url_crawler.fetch_content_from_url(url)
                success = await self.process_single_url_optimized(url, content)
                results.append({"url": url, "success": success})
            except Exception as e:
                logger.error(f"Failed to process URL {url}: {str(e)}")
                results.append({"url": url, "success": False, "error": str(e)})

        self.stats.processing_time_ms = (time.time() - start_time) * 1000

        return {
            "results": results,
            "stats": {
                "total_processed": len([r for r in results if r["success"]]),
                "total_failed": len([r for r in results if not r["success"]]),
                "total_urls": len(urls),
                "processing_time_ms": self.stats.processing_time_ms,
                "average_time_per_url_ms": self.stats.processing_time_ms / len(urls) if urls else 0,
                "total_tokens_processed": self.stats.total_tokens,
                "tokens_per_second": self.stats.total_tokens / (self.stats.processing_time_ms / 1000) if self.stats.processing_time_ms > 0 else 0
            }
        }

    def get_processing_stats(self) -> ProcessStats:
        """Get current processing statistics"""
        return self.stats

    def reset_stats(self):
        """Reset processing statistics"""
        self.stats = ProcessStats()