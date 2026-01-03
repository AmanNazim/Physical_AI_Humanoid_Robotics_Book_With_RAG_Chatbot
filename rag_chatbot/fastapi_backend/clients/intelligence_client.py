from typing import List, Dict, Any, Optional
import httpx
from ..config import settings
from ..utils.logging import get_logger
from ..models.response_models import Source

logger = get_logger(__name__)


class IntelligenceClient:
    """
    Intelligence client wrapper for the FastAPI backend.
    This client provides a clean interface to interact with the Intelligence (LLM) subsystem.
    """

    def __init__(self):
        self.api_key = settings.llm_api_key
        self.base_url = settings.llm_base_url
        self.model = settings.llm_model
        self.timeout = httpx.Timeout(60.0)  # 60 seconds timeout

    async def generate_response(
        self,
        query: str,
        context_chunks: List[Source],
        session_id: Optional[str] = None
    ) -> str:
        """
        Generate a response using the intelligence subsystem (LLM).

        Args:
            query: User's query
            context_chunks: List of context chunks to provide to the LLM
            session_id: Optional session ID for conversation context

        Returns:
            Generated response string
        """
        try:
            # Prepare the context from the source chunks
            context_text = "\n\n".join([chunk.text for chunk in context_chunks])

            # Prepare the messages for the LLM
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the provided context. If the context doesn't contain the information needed to answer the question, say so."
                },
                {
                    "role": "user",
                    "content": f"Context: {context_text}\n\nQuestion: {query}\n\nPlease provide a detailed answer based on the context provided. If the context doesn't contain the information needed to answer the question, say so."
                }
            ]

            # Prepare the request payload
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1000,
                "stream": False  # For now, not streaming
            }

            # Make the request to the LLM API
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers
                )

                if response.status_code == 200:
                    result = response.json()
                    answer = result["choices"][0]["message"]["content"].strip()
                    logger.info(f"Generated response for query: {query[:50]}...")
                    return answer
                else:
                    logger.error(f"LLM API error: {response.status_code} - {response.text}")
                    raise Exception(f"LLM API error: {response.status_code}")

        except Exception as e:
            logger.error(f"Error generating response from intelligence subsystem: {str(e)}")
            raise

    async def generate_streaming_response(
        self,
        query: str,
        context_chunks: List[Source],
        session_id: Optional[str] = None
    ):
        """
        Generate a streaming response using the intelligence subsystem (LLM).

        Args:
            query: User's query
            context_chunks: List of context chunks to provide to the LLM
            session_id: Optional session ID for conversation context

        Yields:
            Streamed tokens/fragments of the response
        """
        try:
            # Prepare the context from the source chunks
            context_text = "\n\n".join([chunk.text for chunk in context_chunks])

            # Prepare the messages for the LLM
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the provided context. If the context doesn't contain the information needed to answer the question, say so."
                },
                {
                    "role": "user",
                    "content": f"Context: {context_text}\n\nQuestion: {query}\n\nPlease provide a detailed answer based on the context provided. If the context doesn't contain the information needed to answer the question, say so."
                }
            ]

            # Prepare the request payload
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1000,
                "stream": True  # Enable streaming
            }

            # Make the request to the LLM API
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status_code == 200:
                        async for chunk in response.aiter_lines():
                            if chunk.startswith("data: "):
                                data = chunk[6:]  # Remove "data: " prefix
                                if data and data != "[DONE]":
                                    yield data
                    else:
                        logger.error(f"Streaming LLM API error: {response.status_code} - {await response.aread()}")
                        raise Exception(f"Streaming LLM API error: {response.status_code}")

        except Exception as e:
            logger.error(f"Error generating streaming response from intelligence subsystem: {str(e)}")
            raise

    async def validate_connection(self) -> bool:
        """
        Validate the connection to the intelligence subsystem.

        Returns:
            bool: True if connection is valid
        """
        try:
            # Test with a simple query
            test_response = await self.generate_response(
                "Hello, can you confirm you're working?",
                []
            )
            return len(test_response) > 0
        except Exception as e:
            logger.error(f"Intelligence client validation failed: {str(e)}")
            return False