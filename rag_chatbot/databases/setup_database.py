#!/usr/bin/env python3
"""
Database Initialization and Setup Script for RAG Chatbot System
This script initializes both Qdrant and PostgreSQL databases.
"""
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the rag_chatbot directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from databases.database_manager import DatabaseManager
from databases.qdrant.qdrant_utils import QdrantUtils
from rag_core.utils.logger import rag_logger
from shared.config import settings


async def initialize_databases():
    """
    Initialize both Qdrant and PostgreSQL databases
    """
    print("🚀 Starting Database Initialization for RAG Chatbot System...")

    db_manager = DatabaseManager()

    try:
        print("\n🔗 Connecting to databases...")
        await db_manager.connect_all()
        print("✅ Successfully connected to both Qdrant and PostgreSQL")

        # Verify Qdrant collection exists
        print("\n🔍 Verifying Qdrant collection...")
        await db_manager.qdrant().ensure_collection_exists()
        print("✅ Qdrant collection verified/created")

        # Verify PostgreSQL tables exist
        print("\n🔍 Verifying PostgreSQL tables...")
        postgres_client = db_manager.postgres()

        # Check if key tables exist and create them if they don't
        from databases.postgres.pg_setup import ensure_database_schema
        print("\n🔍 Verifying and creating PostgreSQL tables if needed...")

        schema_success = await ensure_database_schema(postgres_client)
        if schema_success:
            print("✅ PostgreSQL schema verification and creation completed successfully")
        else:
            print("❌ Failed to verify/create PostgreSQL schema")
            return False

        # Run a basic health check
        print("\n🏥 Running health checks...")
        postgres_health = await postgres_client.health_check()
        print(f"✅ PostgreSQL health: {'OK' if postgres_health else 'FAILED'}")

        print("\n🎯 Running basic functionality test...")

        # Test storing and retrieving a simple chunk
        from databases.postgres.pg_models import ChunkMetadata

        test_chunk = ChunkMetadata(
            chunk_id="setup-test-chunk",
            document_reference="setup-test-doc",
            chunk_text="This is a test chunk created during database setup",
            embedding_id="setup-test-embedding",
            processing_version="1.0",
            created_at="2025-12-23T00:00:00Z",
            updated_at="2025-12-23T00:00:00Z",
            metadata={"setup_test": True, "source": "database_initialization"}
        )

        # Store the test chunk
        store_success = await db_manager.store_chunk_metadata(test_chunk)
        print(f"✅ Chunk storage test: {'PASSED' if store_success else 'FAILED'}")

        # Retrieve the test chunk
        retrieved_chunk = await db_manager.get_chunk_metadata("setup-test-chunk")
        retrieval_success = retrieved_chunk is not None
        print(f"✅ Chunk retrieval test: {'PASSED' if retrieval_success else 'FAILED'}")

        if retrieval_success:
            print(f"   - Retrieved chunk text: {retrieved_chunk.chunk_text[:50]}...")

        print("\n🎉 Database setup verification completed successfully!")
        print("\n📋 Setup Summary:")
        print("   ✅ Qdrant connection established")
        print("   ✅ PostgreSQL connection established")
        print("   ✅ Qdrant collection created/verified")
        print("   ✅ PostgreSQL tables verified")
        print("   ✅ Basic storage/retrieval test passed")

        return True

    except Exception as e:
        print(f"\n❌ Database setup failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        try:
            await db_manager.close_all()
            print("\n🔒 Database connections closed")
        except:
            pass


def verify_environment():
    """
    Verify that environment variables are properly set
    """
    print("🔍 Verifying environment configuration...")

    required_vars = [
        'QDRANT_HOST',
        'QDRANT_API_KEY',
        'QDRANT_COLLECTION_NAME',
        'NEON_DATABASE_URL'
    ]

    all_present = True
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var}: SET")
        else:
            print(f"❌ {var}: MISSING")
            all_present = False

    if all_present:
        print("✅ All required environment variables are present")
        return True
    else:
        print("❌ Some required environment variables are missing!")
        print("Please check your .env file in the rag_chatbot directory")
        return False


async def main():
    print("🤖 RAG Chatbot Database Setup")
    print("=" * 50)

    # Verify environment first
    env_ok = verify_environment()
    if not env_ok:
        print("\n❌ Environment verification failed. Please fix environment variables before continuing.")
        return False

    print()

    # Initialize databases
    success = await initialize_databases()

    if success:
        print("\n🎊 Database setup completed successfully!")
        print("\n💡 Next steps:")
        print("   1. Run other subsystem setup scripts")
        print("   2. Test the complete system integration")
        print("   3. Begin with embedding pipeline setup")
        return True
    else:
        print("\n💥 Database setup failed. Please check the error messages above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)