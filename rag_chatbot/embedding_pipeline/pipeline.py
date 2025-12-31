"""
Main pipeline orchestration for embeddings generation
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib
import time

from .config import validate_config
from .base_classes import Chunk
from .url_crawler import URLCrawler, SitemapParser
from .file_processor import FileProcessor
from .text_preprocessor import TextPreprocessor
from .chunking_engine import ChunkProcessor
from .gemini_client import EmbeddingProcessor
from .database import DatabaseManager
from .reembedding import SelectiveReembedder

logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    """Main orchestration class for the embeddings pipeline"""

    def __init__(self):
        # Validate configuration at startup
        validate_config()

        # Initialize all components
        self.url_crawler = URLCrawler()
        self.file_processor = FileProcessor()
        self.text_preprocessor = TextPreprocessor()
        self.chunk_processor = ChunkProcessor()
        self.embedding_processor = EmbeddingProcessor()
        self.database_manager = DatabaseManager()
        self.selective_reembedder = SelectiveReembedder(self)

        # Processing statistics
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'embeddings_stored': 0,
            'total_processing_time': 0,
            'start_time': None
        }

    async def initialize(self):
        """Initialize all components that require async setup"""
        await self.embedding_processor.initialize()
        await self.database_manager.initialize()

    async def process_from_sitemap(self, sitemap_url: str) -> Dict[str, Any]:
        """
        Process content from sitemap URL - main entry point for URL-based processing

        Args:
            sitemap_url: URL to the sitemap.xml file

        Returns:
            Dictionary with processing results and statistics
        """
        self.stats['start_time'] = time.time()

        logger.info(f"Starting sitemap-based processing from: {sitemap_url}")

        # Step 1: Crawl and get all URLs from sitemap
        urls = await self.url_crawler.get_all_urls_from_sitemap(sitemap_url)
        logger.info(f"Found {len(urls)} URLs to process")

        results = {
            'successful': [],
            'failed': [],
            'stats': {
                'total_urls': len(urls),
                'successful': 0,
                'failed': 0
            }
        }

        # Step 2: Process each URL one by one (as required)
        for i, url_data in enumerate(urls):
            url = url_data['url']
            logger.info(f"Processing URL {i+1}/{len(urls)}: {url}")

            try:
                # Fetch content from URL
                content = await self.url_crawler.fetch_content_from_url(url)

                # Process the content through the pipeline
                url_result = await self.process_content(content, document_reference=url)

                if url_result['success']:
                    results['successful'].append({
                        'url': url,
                        'chunks_processed': url_result['chunks_processed'],
                        'embeddings_generated': url_result['embeddings_generated']
                    })
                    results['stats']['successful'] += 1
                else:
                    results['failed'].append({
                        'url': url,
                        'error': url_result.get('error', 'Unknown error')
                    })
                    results['stats']['failed'] += 1

            except Exception as e:
                logger.error(f"Failed to process URL {url}: {str(e)}")
                results['failed'].append({
                    'url': url,
                    'error': str(e)
                })
                results['stats']['failed'] += 1

        # Calculate final statistics
        processing_time = time.time() - self.stats['start_time']
        self.stats['total_processing_time'] = processing_time

        results['pipeline_stats'] = self.stats

        logger.info(f"Sitemap processing completed. Successful: {results['stats']['successful']}, Failed: {results['stats']['failed']}")

        return results

    async def process_content(self, content: str, document_reference: Optional[str] = None) -> Dict[str, Any]:
        """
        Process content through the full pipeline

        Args:
            content: Text content to process
            document_reference: Reference to the source document

        Returns:
            Dictionary with processing results
        """
        try:
            # Step 1: Preprocess text
            processed_content, validation_errors = self.text_preprocessor.preprocess(content)

            if validation_errors:
                logger.warning(f"Text preprocessing validation errors: {validation_errors}")

            # Step 2: Chunk the content
            chunks = self.chunk_processor.chunk_text(processed_content, document_reference)

            logger.info(f"Created {len(chunks)} chunks from content")

            # Update stats
            self.stats['chunks_created'] += len(chunks)

            # Step 3: Generate embeddings for all chunks
            embeddings = await self.embedding_processor.generate_embeddings(chunks)

            logger.info(f"Generated {len(embeddings)} embeddings")

            # Update stats
            self.stats['embeddings_generated'] += len(embeddings)

            # Step 4: Verify that we have embeddings for all chunks
            if len(embeddings) != len(chunks):
                logger.warning(f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings")

            # Step 5: Store embeddings to database
            storage_success = await self.database_manager.store_embeddings_with_metadata(chunks, embeddings)

            if storage_success:
                self.stats['embeddings_stored'] += len(embeddings)
                logger.info(f"Successfully stored {len(embeddings)} embeddings to database")
            else:
                logger.error("Failed to store embeddings to database")
                return {
                    'success': False,
                    'error': "Failed to store embeddings to database"
                }

            return {
                'success': True,
                'chunks_processed': len(chunks),
                'embeddings_generated': len(embeddings),
                'embeddings_stored': len(embeddings),
                'validation_errors': validation_errors
            }

        except Exception as e:
            logger.error(f"Error processing content: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    async def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process content from a file

        Args:
            file_path: Path to the file to process

        Returns:
            Dictionary with processing results
        """
        try:
            # Load content from file
            content = await self.file_processor.load_file(file_path)

            # Process through pipeline
            result = await self.process_content(content, document_reference=file_path)

            # Update stats
            self.stats['documents_processed'] += 1

            return result

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get current pipeline statistics"""
        if self.stats['start_time']:
            self.stats['current_processing_time'] = time.time() - self.stats['start_time']
        return self.stats

    async def selective_reembed_document(self, content: str, document_reference: str) -> Dict[str, Any]:
        """
        Selectively re-embed a document by detecting changes

        Args:
            content: New content to process
            document_reference: Reference to the document

        Returns:
            Dictionary with processing results
        """
        return await self.selective_reembedder.selective_reembed(content, document_reference)

    async def process_batch(self, sources: List[str]) -> Dict[str, Any]:
        """
        Process a batch of sources (files or URLs)

        Args:
            sources: List of file paths or URLs to process

        Returns:
            Dictionary with batch processing results
        """
        results = {
            'successful': [],
            'failed': [],
            'stats': {
                'total_sources': len(sources),
                'successful': 0,
                'failed': 0
            }
        }

        for source in sources:
            if source.startswith(('http://', 'https://')):
                # This is a URL
                result = await self.process_from_url(source)
            else:
                # This is a file path
                result = await self.process_file(source)

            if result['success']:
                results['successful'].append({
                    'source': source,
                    'result': result
                })
                results['stats']['successful'] += 1
            else:
                results['failed'].append({
                    'source': source,
                    'error': result.get('error', 'Unknown error')
                })
                results['stats']['failed'] += 1

        return results

    async def process_from_url(self, url: str) -> Dict[str, Any]:
        """
        Process content from a single URL

        Args:
            url: URL to fetch and process content from

        Returns:
            Dictionary with processing results
        """
        try:
            # Fetch content from URL
            content = await self.url_crawler.fetch_content_from_url(url)

            # Process through pipeline
            result = await self.process_content(content, document_reference=url)

            # Update stats
            self.stats['documents_processed'] += 1

            return result

        except Exception as e:
            logger.error(f"Error processing URL {url}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }


# Convenience function for simple usage
async def generate_embeddings_for_document(document_source: str) -> Dict[str, Any]:
    """
    Convenience function to generate embeddings for a document from file or URL

    Args:
        document_source: File path or URL to process

    Returns:
        Dictionary with processing results
    """
    pipeline = EmbeddingPipeline()
    await pipeline.initialize()

    if document_source.startswith(('http://', 'https://')):
        return await pipeline.process_from_url(document_source)
    else:
        return await pipeline.process_file(document_source)


# Convenience function for sitemap processing
async def generate_embeddings_from_sitemap(sitemap_url: str) -> Dict[str, Any]:
    """
    Convenience function to generate embeddings for all content from a sitemap

    Args:
        sitemap_url: URL to the sitemap.xml file

    Returns:
        Dictionary with processing results
    """
    pipeline = EmbeddingPipeline()
    await pipeline.initialize()

    return await pipeline.process_from_sitemap(sitemap_url)