"""
Chat endpoints for the RAG Chatbot API.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Any, Optional
from pydantic import BaseModel
from ..services.rag_service import RAGService
from ..services.streaming_service import StreamingService
from ..schemas.chat import ChatRequest, ChatResponse
from ..utils.logger import rag_logger
from fastapi.responses import StreamingResponse
import json

router = APIRouter()

# Create a global instance of StreamingService for WebSocket handling
streaming_service = StreamingService()


@router.post("/chat")
async def chat_endpoint(request: ChatRequest) -> Dict[str, Any]:
    """
    Main RAG endpoint and orchestrator.
    1. Accept user query
    2. Store user message in Postgres (not implemented in this version)
    3. Perform retrieval via Qdrant
    4. Call RAG service abstraction (LLM placeholder)
    5. Stream response back (SSE) - for non-streaming requests
    6. Store assistant message (not implemented in this version)
    """
    try:
        rag_service = RAGService()
        await rag_service.initialize()

        # Generate response using RAG approach
        result = await rag_service.generate_response(
            query=request.query,
            top_k=request.max_context,
            session_id=request.session_id
        )

        return {
            "answer": result["answer"],
            "sources": [source.model_dump() for source in result.get("sources", [])],
            "session_id": request.session_id,
            "query": request.query
        }

    except Exception as e:
        rag_logger.error(f"Chat endpoint error: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "answer": "An error occurred while processing your request."
        }


@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Streaming chat endpoint.
    Returns a streaming response with Server-Sent Events.
    """
    try:
        rag_service = RAGService()
        await rag_service.initialize()

        # Stream response using RAG approach with actual token-by-token streaming
        async def generate_stream():
            async for chunk in rag_service.stream_response(
                query=request.query,
                top_k=request.max_context,
                session_id=request.session_id
            ):
                yield chunk

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    except Exception as e:
        rag_logger.error(f"Chat stream endpoint error: {str(e)}")

        async def error_stream():
            error_data = {
                "type": "error",
                "message": str(e)
            }
            yield f"data: {json.dumps(error_data)}\n\n"

        return StreamingResponse(error_stream(), media_type="text/plain")


@router.websocket("/chat/ws/{session_id}")
async def websocket_chat_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time chat streaming.
    Provides bidirectional communication for real-time responses.
    """
    try:
        # Handle the WebSocket connection using the streaming service
        await streaming_service.websocket_handler(websocket, session_id)
    except WebSocketDisconnect:
        rag_logger.info(f"WebSocket disconnected for session: {session_id}")
    except Exception as e:
        rag_logger.error(f"WebSocket error for session {session_id}: {str(e)}")
        try:
            await websocket.close()
        except:
            pass  # Ignore errors when trying to close the websocket