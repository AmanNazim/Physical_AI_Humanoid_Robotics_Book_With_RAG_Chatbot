"""
Hugging Face Spaces interface for the RAG Chatbot API.
This file allows the FastAPI backend to run in Hugging Face Spaces environment.
"""
import os
import logging

# Set up basic logging early
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_app_with_error_handling():
    """
    Create the FastAPI app with error handling for missing dependencies.
    This prevents the app from crashing during startup if external services aren't available.
    """
    try:
        from backend.main import app
        logging.info("Successfully imported backend.main")
        return app
    except ImportError as e:
        logging.error(f"Failed to import backend.main: {e}")
        # Create a minimal app that just serves health checks
        from fastapi import FastAPI
        app = FastAPI(title="RAG Chatbot API (Minimal)", description="Minimal API for health checks")

        @app.get("/")
        async def root():
            return {"message": "RAG Chatbot API is running (minimal mode)"}

        @app.get("/health")
        async def health():
            return {"status": "starting", "service": "RAG Chatbot API"}

        @app.get("/api/v1/health")
        async def api_health():
            return {"status": "starting", "service": "RAG Chatbot API"}

        return app
    except Exception as e:
        logging.error(f"Unexpected error importing backend: {e}")
        from fastapi import FastAPI
        app = FastAPI(title="RAG Chatbot API (Error Mode)", description="API in error recovery mode")

        @app.get("/")
        async def root():
            return {"message": "RAG Chatbot API (Error Recovery Mode)", "error": str(e)}

        @app.get("/health")
        async def health():
            return {"status": "error", "service": "RAG Chatbot API", "error": str(e)}

        return app

# Hugging Face Spaces requires the application to be available as "app"
app = create_app_with_error_handling()

if __name__ == "__main__":
    # Hugging Face Spaces provides the PORT environment variable
    port = int(os.environ.get("PORT", 7860))  # Default to 7860 which is standard for HF Spaces

    import uvicorn
    # Run the FastAPI application
    logging.info(f"Starting server on port {port}")
    uvicorn.run(
        app,  # Pass the actual app instance
        host="0.0.0.0",
        port=port,
        log_level="info",
        timeout_keep_alive=30,
        lifespan="off"  # Disable lifespan to prevent startup hangs
    )