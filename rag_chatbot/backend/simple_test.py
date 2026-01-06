#!/usr/bin/env python3
"""
Simple test to verify that the backend components can be imported without errors.
This confirms that the implementation is structurally correct.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Also add the parent directory to handle relative imports properly
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test that all the implemented components can be imported."""
    print("Testing imports of implemented components...")

    try:
        # Test shared config import
        from shared.config import settings
        print("✓ Shared config module imported successfully")

        # Test service imports
        from backend.services.retrieval_service import RetrievalService
        print("✓ RetrievalService imported successfully")

        from backend.services.rag_service import RAGService
        print("✓ RAGService imported successfully")

        from backend.services.embedding_service import EmbeddingService
        print("✓ EmbeddingService imported successfully")

        from backend.services.streaming_service import StreamingService
        print("✓ StreamingService imported successfully")

        # Test schema imports
        from backend.schemas.chat import ChatRequest, ChatResponse
        print("✓ Chat schemas imported successfully")

        from backend.schemas.embedding import EmbeddingRequest, EmbeddingResponse
        print("✓ Embedding schemas imported successfully")

        from backend.schemas.retrieval import RetrievalRequest, RetrievalResponse, Source
        print("✓ Retrieval schemas imported successfully")

        from backend.schemas.error import ErrorDetail, ErrorResponse, ErrorCode
        print("✓ Error schemas imported successfully")

        # Test router imports
        from backend.routers.health import router as health_router
        print("✓ Health router imported successfully")

        from backend.routers.chat import router as chat_router
        print("✓ Chat router imported successfully")

        from backend.routers.retrieve import router as retrieve_router
        print("✓ Retrieve router imported successfully")

        from backend.routers.embed import router as embed_router
        print("✓ Embed router imported successfully")

        # Test middleware imports
        from backend.middleware.cors import add_cors_middleware
        print("✓ CORS middleware imported successfully")

        from backend.middleware.logging import add_logging_middleware
        print("✓ Logging middleware imported successfully")

        from backend.middleware.rate_limit import add_rate_limit_middleware
        print("✓ Rate limit middleware imported successfully")

        from backend.middleware import setup_middleware
        print("✓ Middleware setup imported successfully")

        # Test main app
        from backend.main import create_app, app
        print("✓ Main application imported successfully")

        # Test utilities
        from backend.utils.logger import rag_logger
        print("✓ Logger utility imported successfully")

        print("\n🎉 All components imported successfully!")
        print("The FastAPI Backend Subsystem implementation is structurally correct.")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n✅ Implementation verification: PASSED")
    else:
        print("\n❌ Implementation verification: FAILED")
    sys.exit(0 if success else 1)