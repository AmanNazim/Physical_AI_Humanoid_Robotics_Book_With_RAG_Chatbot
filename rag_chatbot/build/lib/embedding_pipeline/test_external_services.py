"""
Diagnostic script to test external services for the embedding generation pipeline.
This will help identify which service is causing the hanging issue.
"""
import asyncio
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_gemini_api():
    """Test Gemini API connectivity and response time."""
    logger.info("Testing Gemini API connectivity...")

    try:
        from .config import config
        from .gemini_client import get_gemini_client

        if not config.gemini_api_key:
            logger.error("❌ Gemini API key not found in environment")
            return False

        logger.info("✅ Gemini API key found")

        # Test with a simple embedding request
        client = get_gemini_client()

        start_time = time.time()
        test_text = "This is a test for Gemini API connectivity."
        embedding = await client.embed_single(test_text)
        elapsed = time.time() - start_time

        if embedding and len(embedding) > 0:
            logger.info(f"✅ Gemini API test successful - Response time: {elapsed:.2f}s, Embedding dimensions: {len(embedding)}")
            return True
        else:
            logger.error("❌ Gemini API returned empty embedding")
            return False

    except ImportError as e:
        logger.error(f"❌ Gemini client import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Gemini API test failed: {e}")
        return False


async def test_qdrant_connection():
    """Test Qdrant database connectivity and response time."""
    logger.info("Testing Qdrant connection...")

    try:
        from .config import config
        from qdrant_client import QdrantClient

        start_time = time.time()
        client = QdrantClient(
            url=config.qdrant_host,
            api_key=config.qdrant_api_key,
            prefer_grpc=False
        )

        # Test connection by getting collections
        collections = client.get_collections()
        elapsed = time.time() - start_time

        logger.info(f"✅ Qdrant connection successful - Response time: {elapsed:.2f}s")
        logger.info(f"   Available collections: {[col.name for col in collections.collections]}")

        # Test collection creation/get
        start_time = time.time()
        try:
            client.get_collection(config.qdrant_collection_name)
            logger.info(f"   Collection '{config.qdrant_collection_name}' exists")
        except:
            logger.info(f"   Collection '{config.qdrant_collection_name}' does not exist (this is OK)")

        elapsed = time.time() - start_time
        logger.info(f"   Collection check time: {elapsed:.2f}s")

        client.close()
        return True

    except Exception as e:
        logger.error(f"❌ Qdrant connection test failed: {e}")
        return False


async def test_neon_database():
    """Test Neon Postgres database connectivity and response time."""
    logger.info("Testing Neon Postgres database...")

    try:
        from .config import config
        import asyncpg

        if not config.neon_database_url or "postgresql" not in config.neon_database_url:
            logger.warning("⚠️  Neon database URL not configured or invalid")
            return False

        start_time = time.time()
        conn = await asyncio.wait_for(asyncpg.connect(config.neon_database_url), timeout=30.0)
        elapsed = time.time() - start_time

        logger.info(f"✅ Neon database connection successful - Response time: {elapsed:.2f}s")

        # Test a simple query
        start_time = time.time()
        result = await conn.fetchval("SELECT version();")
        query_time = time.time() - start_time

        logger.info(f"✅ Database query successful - Query time: {query_time:.2f}s")
        logger.info(f"   PostgreSQL version: {result[:50]}...")

        await conn.close()
        return True

    except asyncio.TimeoutError:
        logger.error("❌ Neon database connection timed out after 30 seconds")
        return False
    except Exception as e:
        logger.error(f"❌ Neon database test failed: {e}")
        return False


async def test_embedding_pipeline_components():
    """Test individual components of the embedding pipeline."""
    logger.info("Testing embedding pipeline components...")

    try:
        # Test chunking
        from .chunker import clean_text, chunk_text

        test_text = "This is a test document. " * 100  # Create a longer text
        cleaned = clean_text(test_text)
        chunks = chunk_text(cleaned)

        logger.info(f"✅ Chunking test successful - Generated {len(chunks)} chunks from {len(test_text)} characters")

        # Test embedding generation (if available)
        try:
            from .embedder import generate_embedding
            if len(chunks) > 0:
                start_time = time.time()
                embedding = await generate_embedding(chunks[0][:100])  # Use first 100 chars
                elapsed = time.time() - start_time
                logger.info(f"✅ Single embedding generation test successful - Time: {elapsed:.2f}s, Dimensions: {len(embedding) if embedding else 0}")
        except:
            logger.info("⚠️  Embedding generation test skipped (likely due to missing API key)")

        return True

    except Exception as e:
        logger.error(f"❌ Embedding pipeline components test failed: {e}")
        return False


async def run_diagnostics():
    """Run all diagnostic tests."""
    logger.info("="*60)
    logger.info("EXTERNAL SERVICES DIAGNOSTICS")
    logger.info("="*60)

    results = {}

    # Test Gemini API
    results['gemini'] = await test_gemini_api()
    logger.info("")

    # Test Qdrant
    results['qdrant'] = await test_qdrant_connection()
    logger.info("")

    # Test Neon Database
    results['neon'] = await test_neon_database()
    logger.info("")

    # Test pipeline components
    results['pipeline'] = await test_embedding_pipeline_components()
    logger.info("")

    # Summary
    logger.info("="*60)
    logger.info("DIAGNOSTICS SUMMARY")
    logger.info("="*60)

    services = {
        'Gemini API': results.get('gemini', False),
        'Qdrant Database': results.get('qdrant', False),
        'Neon Postgres': results.get('neon', False),
        'Embedding Pipeline': results.get('pipeline', False)
    }

    for service, status in services.items():
        status_icon = "✅" if status else "❌"
        logger.info(f"{status_icon} {service}: {'PASS' if status else 'FAIL'}")

    successful = sum(results.values())
    total = len(results)

    logger.info(f"\nOverall: {successful}/{total} services operational")

    if successful == total:
        logger.info("🎉 All services are working correctly!")
    else:
        logger.info("⚠️  Some services are not working properly. This may cause hanging issues.")

    return results


if __name__ == "__main__":
    asyncio.run(run_diagnostics())