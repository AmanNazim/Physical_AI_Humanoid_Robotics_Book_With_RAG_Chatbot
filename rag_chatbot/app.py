"""
Hugging Face Spaces interface for the RAG Chatbot API.
This file allows the FastAPI backend to run in Hugging Face Spaces environment.
"""
import os
import uvicorn
from backend.main import app

# Hugging Face Spaces requires the application to be available as "app"
# The main FastAPI app is already defined in backend.main

if __name__ == "__main__":
    # Hugging Face Spaces provides the PORT environment variable
    port = int(os.environ.get("PORT", 8000))

    # Run the FastAPI application
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )