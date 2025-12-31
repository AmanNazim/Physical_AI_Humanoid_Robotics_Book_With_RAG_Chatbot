#!/usr/bin/env python3
"""
Database Migration Script
This script handles database schema creation and migrations.
"""
import asyncio
import sys
from pathlib import Path

# Add the rag_chatbot directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from databases.postgres.pg_setup import initialize_database_schema
from rag_core.utils.logger import rag_logger


async def main():
    """
    Main function to run database migrations
    """
    print("🚀 Starting database schema initialization...")

    try:
        success = await initialize_database_schema()

        if success:
            print("\n✅ Database schema initialization completed successfully!")
            return True
        else:
            print("\n❌ Database schema initialization failed!")
            return False

    except Exception as e:
        print(f"\n💥 Error during database schema initialization: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)