#!/usr/bin/env python3
"""
Entry point script for the embeddings pipeline
"""
import asyncio
import argparse
import sys
import os
import logging
from pathlib import Path

# Add the rag_chatbot directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from embedding_pipeline import EmbeddingPipeline, generate_embeddings_for_document, generate_embeddings_from_sitemap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="Embeddings Pipeline for RAG Chatbot")
    parser.add_argument("source", help="Document source (file path or URL)")
    parser.add_argument("--sitemap", action="store_true", help="Process as sitemap URL")
    parser.add_argument("--file", action="store_true", help="Process as file path")
    parser.add_argument("--output", "-o", help="Output directory for results")

    args = parser.parse_args()

    logger.info(f"Starting embeddings pipeline for source: {args.source}")

    # Create pipeline instance
    pipeline = EmbeddingPipeline()
    logger.info("Initializing pipeline...")
    await pipeline.initialize()
    logger.info("Pipeline initialized successfully")

    try:
        if args.sitemap:
            logger.info(f"Processing sitemap from: {args.source}")
            logger.info("Starting sitemap crawling and embedding generation...")
            result = await generate_embeddings_from_sitemap(args.source)
            logger.info("Sitemap processing completed")
        else:
            logger.info(f"Processing document from: {args.source}")
            logger.info("Starting document processing and embedding generation...")
            result = await generate_embeddings_for_document(args.source)
            logger.info("Document processing completed")

        logger.info(f"Processing completed with result: {result}")

        # Print final statistics
        stats = pipeline.get_pipeline_stats()
        logger.info(f"Final pipeline statistics: {stats}")
        print(f"Final statistics: {stats}")

        # Log detailed summary
        logger.info("=== EMBEDDING GENERATION AND STORAGE SUMMARY ===")
        logger.info(f"Source processed: {args.source}")
        logger.info(f"Documents processed: {stats.get('documents_processed', 'N/A')}")
        logger.info(f"Chunks created: {stats.get('chunks_created', 'N/A')}")
        logger.info(f"Embeddings generated: {stats.get('embeddings_generated', 'N/A')}")
        logger.info(f"Embeddings stored: {stats.get('embeddings_stored', 'N/A')}")
        logger.info(f"Total processing time: {stats.get('total_processing_time', 'N/A')} seconds")
        logger.info(f"Embedding dimension: {pipeline.embedding_processor.output_dimensionality}")

        # Handle success/failure statistics based on processing type
        if args.sitemap:
            # For sitemap processing, we have detailed success/failure breakdown
            successful_count = result.get('stats', {}).get('successful', 'N/A')
            failed_count = result.get('stats', {}).get('failed', 'N/A')
            total_urls = result.get('stats', {}).get('total_urls', 'N/A')

            successful_list = result.get('successful', [])
            failed_list = result.get('failed', [])

            logger.info(f"Total documents attempted: {total_urls}")
            logger.info(f"Documents processed successfully: {successful_count}")
            logger.info(f"Documents failed: {failed_count}")

            if successful_list:
                logger.info(f"Successfully processed documents ({len(successful_list)}):")
                for item in successful_list[:5]:  # Show first 5 successful items
                    logger.info(f"  - {item.get('url', 'N/A')} (chunks: {item.get('chunks_processed', 'N/A')}, embeddings: {item.get('embeddings_generated', 'N/A')})")
                if len(successful_list) > 5:
                    logger.info(f"  ... and {len(successful_list) - 5} more")

            if failed_list:
                logger.info(f"Failed documents ({len(failed_list)}):")
                for item in failed_list[:5]:  # Show first 5 failed items
                    logger.info(f"  - {item.get('url', 'N/A')}: {item.get('error', 'Unknown error')}")
                if len(failed_list) > 5:
                    logger.info(f"  ... and {len(failed_list) - 5} more")

            if successful_count != 'N/A' and total_urls != 'N/A':
                success_rate = (successful_count / total_urls * 100) if total_urls > 0 else 0
                logger.info(f"Success rate: {success_rate:.2f}%")
        else:
            # For single document processing, check success/failure from the result
            success = result.get('success', False)
            error = result.get('error', None)

            logger.info(f"Document processing status: {'SUCCESS' if success else 'FAILED'}")

            if success:
                logger.info(f"Chunks processed: {result.get('chunks_processed', 'N/A')}")
                logger.info(f"Embeddings generated: {result.get('embeddings_generated', 'N/A')}")
            else:
                logger.info(f"Error: {error}")

        logger.info("=== PROCESSING COMPLETE ===")

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())