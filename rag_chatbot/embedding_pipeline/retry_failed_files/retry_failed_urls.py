#!/usr/bin/env python3
"""
Retry mechanism for failed URLs from the embedding pipeline
"""
import asyncio
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import logging

# Add the rag_chatbot directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from embedding_pipeline import EmbeddingPipeline
from parse_failed_urls import load_urls_from_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def retry_failed_urls(urls: List[str], max_retries: int = 3) -> Dict[str, Any]:
    """
    Retry processing for failed URLs.

    Args:
        urls: List of URLs to retry
        max_retries: Maximum number of retry attempts per URL

    Returns:
        Dictionary with results of retry attempts
    """
    # Initialize the pipeline
    pipeline = EmbeddingPipeline()
    await pipeline.initialize()

    results = {
        'successful': [],
        'failed': [],
        'stats': {
            'total': len(urls),
            'successful': 0,
            'failed': 0
        }
    }

    for i, url in enumerate(urls, 1):
        logger.info(f"Retrying URL {i}/{len(urls)}: {url}")

        success = False
        error_msg = ""

        for attempt in range(max_retries):
            try:
                logger.info(f"  Attempt {attempt + 1}/{max_retries}")

                # Process the URL using the pipeline
                result = await pipeline.process_from_url(url)

                if result.get('success', False):
                    success = True
                    results['successful'].append({
                        'url': url,
                        'attempt': attempt + 1,
                        'chunks_processed': result.get('chunks_processed', 0),
                        'embeddings_generated': result.get('embeddings_generated', 0),
                        'embeddings_stored': result.get('embeddings_stored', 0)
                    })
                    logger.info(f"  SUCCESS: {url}")
                    break
                else:
                    error_msg = result.get('error', 'Unknown error')
                    logger.warning(f"  Attempt {attempt + 1} failed: {error_msg}")

            except Exception as e:
                error_msg = str(e)
                logger.warning(f"  Attempt {attempt + 1} failed with exception: {error_msg}")

            if not success and attempt < max_retries - 1:
                # Wait before retry with exponential backoff
                wait_time = 2 ** attempt  # 2, 4, 8 seconds...
                logger.info(f"  Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)

        if not success:
            results['failed'].append({
                'url': url,
                'attempts': max_retries,
                'error': error_msg
            })
            logger.error(f"  FAILED after {max_retries} attempts: {url} - {error_msg}")

    results['stats']['successful'] = len(results['successful'])
    results['stats']['failed'] = len(results['failed'])

    return results

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Retry failed URLs from embedding pipeline")
    parser.add_argument("input_file", help="File containing failed URLs (one per line)")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retry attempts (default: 3)")
    parser.add_argument("--output-file", default="retry_results.json", help="Output file for results (JSON format)")

    args = parser.parse_args()

    # Load failed URLs
    logger.info(f"Loading failed URLs from {args.input_file}")
    failed_urls = load_urls_from_file(args.input_file)

    if not failed_urls:
        logger.info("No failed URLs found in the input file.")
        return

    logger.info(f"Found {len(failed_urls)} URLs to retry")

    # Perform retries
    logger.info("Starting retry process...")
    results = await retry_failed_urls(failed_urls, max_retries=args.max_retries)

    # Print summary
    logger.info("=== RETRY SUMMARY ===")
    logger.info(f"Total URLs to retry: {results['stats']['total']}")
    logger.info(f"Successfully processed: {results['stats']['successful']}")
    logger.info(f"Still failed: {results['stats']['failed']}")

    if results['successful']:
        logger.info("\nSuccessfully processed URLs:")
        for item in results['successful']:
            logger.info(f"  - {item['url']} (attempt {item['attempt']})")

    if results['failed']:
        logger.info("\nStill failed URLs:")
        for item in results['failed']:
            logger.info(f"  - {item['url']}: {item['error']}")

    # Save results to JSON file
    import json
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    asyncio.run(main())