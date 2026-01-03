from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Any
import json
from ...services.query_service import QueryService
from ...services.streaming_service import StreamingService
from ...services.rag_service import RAGService
from ...utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["websocket"])

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[WebSocket, str] = {}

    async def connect(self, websocket: WebSocket, client_id: str = None):
        await websocket.accept()
        self.active_connections[websocket] = client_id or str(id(websocket))

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            del self.active_connections[websocket]

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections.keys():
            await connection.send_text(message)


manager = ConnectionManager()


@router.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for streaming chat responses.
    """
    await manager.connect(websocket)
    try:
        query_service = QueryService()
        await query_service.initialize()

        rag_service = RAGService()
        streaming_service = StreamingService()

        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message_data = json.loads(data)

                # Extract query and other parameters
                query = message_data.get("query", "")
                session_id = message_data.get("session_id", "")
                max_context = message_data.get("max_context", 5)

                if not query:
                    await manager.send_personal_message(
                        json.dumps({"type": "error", "message": "Query is required"}),
                        websocket
                    )
                    continue

                # Retrieve context
                sources = await query_service.retrieve_context(query, max_context)

                # Stream the response
                async for chunk in rag_service.generate_streaming_response(query, sources, session_id):
                    try:
                        # Send chunk to client
                        await manager.send_personal_message(chunk, websocket)
                    except Exception as e:
                        logger.error(f"Error sending chunk: {str(e)}")
                        break

                # Send completion message
                await manager.send_personal_message(
                    json.dumps({"type": "complete", "message": "Response completed"}),
                    websocket
                )

            except WebSocketDisconnect:
                manager.disconnect(websocket)
                logger.info(f"WebSocket disconnected: {manager.active_connections.get(websocket)}")
                break
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}")
                await manager.send_personal_message(
                    json.dumps({"type": "error", "message": str(e)}),
                    websocket
                )
                break
    except Exception as e:
        logger.error(f"WebSocket connection error: {str(e)}")
    finally:
        manager.disconnect(websocket)