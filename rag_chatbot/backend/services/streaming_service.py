"""
Service for handling streaming responses.
Provides utilities for streaming tokens and managing streaming sessions.
"""
from typing import AsyncGenerator, Dict, Any
from fastapi import WebSocket
from ..utils.logger import rag_logger
from ..schemas.retrieval import Source
import json
import asyncio


class StreamingService:
    """
    Service for handling streaming responses.
    Provides utilities for streaming tokens and managing streaming sessions.
    """

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def stream_response(
        self,
        query: str,
        sources: list,
        answer: str
    ) -> AsyncGenerator[str, None]:
        """
        Stream response tokens using Server-Sent Events (SSE) format.

        Args:
            query: User's query
            sources: List of sources used
            answer: Full answer to stream

        Yields:
            Streamed response chunks in SSE format
        """
        try:
            # Send sources first
            for source in sources:
                source_data = {
                    "type": "source",
                    "chunk_id": source.chunk_id,
                    "document_id": source.document_id,
                    "text": source.text[:200] + "..." if len(source.text) > 200 else source.text,
                    "score": source.score,
                    "metadata": source.metadata
                }
                yield f"data: {json.dumps(source_data)}\n\n"

            # Stream the answer token by token (simulated)
            # In a real implementation, this would stream as the LLM generates tokens
            if answer and answer.strip():  # Only stream if there's a non-empty answer
                # Send the complete answer as one chunk to ensure it appears in the UI
                # But format it properly for the frontend
                chunk_data = {
                    "type": "token",
                    "content": answer,
                    "index": 0,
                    "total_tokens": 1
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
            else:
                # Send a default message if the answer is empty
                chunk_data = {
                    "type": "token",
                    "content": "I'm sorry, I couldn't generate a response for your query. Please try again.",
                    "index": 0,
                    "total_tokens": 1
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"

            # Send completion message
            completion_data = {
                "type": "complete",
                "message": "Response completed"
            }
            yield f"data: {json.dumps(completion_data)}\n\n"

        except Exception as e:
            rag_logger.error(f"Error in streaming response: {str(e)}")
            error_data = {
                "type": "error",
                "message": str(e)
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    async def stream_chat_response(
        self,
        query: str,
        sources: list,
        answer: str
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat response with proper formatting.

        Args:
            query: User's query
            sources: List of sources used
            answer: Full answer to stream

        Yields:
            Formatted chat response chunks
        """
        async for chunk in self.stream_response(query, sources, answer):
            yield chunk

    async def websocket_handler(self, websocket: WebSocket, session_id: str):
        """
        Handle WebSocket connection for real-time streaming.

        Args:
            websocket: WebSocket connection
            session_id: Session ID for the connection
        """
        await websocket.accept()
        self.active_connections[session_id] = websocket

        try:
            # Send initial connection confirmation
            await websocket.send_text(json.dumps({
                "type": "connection",
                "message": "Connected to streaming service",
                "session_id": session_id
            }))

            # Keep connection alive and handle incoming messages
            while True:
                # Wait for messages from client (with timeout)
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=300)  # 5 minute timeout
                    message = json.loads(data)

                    # Handle different message types
                    if message.get("type") == "ping":
                        await websocket.send_text(json.dumps({
                            "type": "pong",
                            "timestamp": message.get("timestamp")
                        }))
                    elif message.get("type") == "disconnect":
                        break
                except asyncio.TimeoutError:
                    # Send heartbeat to keep connection alive
                    await websocket.send_text(json.dumps({
                        "type": "heartbeat",
                        "timestamp": json.dumps({"timestamp": asyncio.get_event_loop().time()})
                    }))
        except Exception as e:
            rag_logger.error(f"WebSocket error in session {session_id}: {str(e)}")
        finally:
            # Clean up connection
            if session_id in self.active_connections:
                del self.active_connections[session_id]
            await websocket.close()

    async def stream_to_websocket(self, session_id: str, query: str, sources: list, answer: str):
        """
        Stream response to a specific WebSocket connection.

        Args:
            session_id: Session ID of the WebSocket connection
            query: User's query
            sources: List of sources used
            answer: Full answer to stream
        """
        if session_id not in self.active_connections:
            rag_logger.warning(f"No active connection for session {session_id}")
            return

        websocket = self.active_connections[session_id]

        try:
            # Send sources first
            for source in sources:
                source_data = {
                    "type": "source",
                    "chunk_id": source.chunk_id,
                    "document_id": source.document_id,
                    "text": source.text[:200] + "..." if len(source.text) > 200 else source.text,
                    "score": source.score,
                    "metadata": source.metadata
                }
                await websocket.send_text(json.dumps(source_data))

            # Stream the answer token by token
            words = answer.split()
            for i, word in enumerate(words):
                chunk_data = {
                    "type": "token",
                    "content": word + (" " if i < len(words) - 1 else ""),
                    "index": i,
                    "total_tokens": len(words)
                }
                await websocket.send_text(json.dumps(chunk_data))

            # Send completion message
            completion_data = {
                "type": "complete",
                "message": "Response completed"
            }
            await websocket.send_text(json.dumps(completion_data))

        except Exception as e:
            rag_logger.error(f"Error streaming to WebSocket {session_id}: {str(e)}")
            error_data = {
                "type": "error",
                "message": str(e)
            }
            await websocket.send_text(json.dumps(error_data))

    async def create_stream_session(self, session_id: str = None) -> str:
        """
        Create a new streaming session.

        Args:
            session_id: Optional session ID to use

        Returns:
            Session ID for the streaming session
        """
        import uuid
        session_id = session_id or str(uuid.uuid4())
        rag_logger.info(f"Created streaming session: {session_id}")
        return session_id

    async def cleanup_stream_session(self, session_id: str):
        """
        Clean up resources for a streaming session.

        Args:
            session_id: Session ID to clean up
        """
        rag_logger.info(f"Cleaning up streaming session: {session_id}")

        # Close WebSocket connection if exists
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            await websocket.close()
            del self.active_connections[session_id]