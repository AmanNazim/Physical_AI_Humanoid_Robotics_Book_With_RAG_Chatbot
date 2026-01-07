#!/usr/bin/env python3
"""
Hugging Face Spaces startup script for the RAG Chatbot API.
This script handles the startup process for Hugging Face Spaces environment.
"""
import os
import sys
import subprocess
import threading
import time
import signal
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_port_availability(port):
    """Check if the port is available by trying to bind to it."""
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        return result != 0  # Port is available if connect_ex returns non-zero
    except Exception:
        return True  # If we can't check, assume it's available

def main():
    # Get the port from environment variable (provided by Hugging Face Spaces)
    port = int(os.environ.get("PORT", 7860))
    logger.info(f"Starting application on port {port}")

    # Verify the port is available
    if not check_port_availability(port):
        logger.warning(f"Port {port} might be in use, attempting to start anyway")

    # Import the app here to defer any heavy initialization
    try:
        from backend.main import app
        logger.info("Successfully imported application")
    except Exception as e:
        logger.error(f"Failed to import application: {e}")
        sys.exit(1)

    # Start the Uvicorn server
    import uvicorn

    try:
        logger.info(f"Starting Uvicorn server on 0.0.0.0:{port}")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            timeout_keep_alive=30,
            lifespan="off"  # Disable lifespan to avoid startup issues
        )
    except Exception as e:
        logger.error(f"Failed to start Uvicorn server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()