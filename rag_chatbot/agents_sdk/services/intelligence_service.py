"""
Intelligence Layer (OpenAI Agents SDK) Service for Global RAG Chatbot System.

This service serves as the cognitive reasoning engine for the entire RAG system.
It processes user queries forwarded by the FastAPI backend, performs sophisticated
reasoning over retrieved context chunks from Qdrant and PostgreSQL, and generates
accurate, coherent, and contextually-aware responses using the OpenAI Agents SDK.
"""

from typing import List, Dict, Any, Optional, AsyncGenerator
import asyncio
import json
import os
from datetime import datetime

# Import Pydantic BaseModel first to avoid reference issues
from pydantic import BaseModel

# Import OpenAI Agents SDK
from agents import Agent, Runner, SQLiteSession, function_tool, input_guardrail, output_guardrail, GuardrailFunctionOutput
from agents.extensions.models.litellm_model import LitellmModel
from agents.model_settings import ModelSettings

# Import event types for streaming - with fallback if direct import fails
# try:
from openai.types.responses import ResponseTextDeltaEvent
# except ImportError:
#     # Define a fallback class if direct import fails
#     class ResponseTextDeltaEvent:
#         def __init__(self, **kwargs):
#             for k, v in kwargs.items():
#                 setattr(self, k, v)

# Import existing modules to maintain integration
from shared.config import settings
from backend.utils.logger import rag_logger
from backend.schemas.retrieval import Source
from backend.services.streaming_service import StreamingService


class BookContext(BaseModel):
    """Model for book context data."""
    query: str
    selected_text: Optional[str] = None
    chunks: List[Dict[str, Any]] = []


class ConversationContext(BaseModel):
    """Model for conversation context data."""
    session_id: str
    max_turns: int = 5
    max_tokens: int = 2000


class ResponseScope(BaseModel):
    """Model for response scope validation."""
    response: str
    context_chunks: List[Dict[str, Any]]
    user_query: str


class IntelligenceService:
    """
    Intelligence Layer Service using OpenAI Agents SDK for reasoning and response generation.

    This service integrates with existing RAGService to receive query-context pairs,
    applies advanced reasoning using OpenAI Agents SDK, and generates grounded responses
    while maintaining constitutional boundaries.
    """

    def __init__(self):
        """Initialize the Intelligence Service with proper configuration."""
        self.settings = settings
        self.logger = rag_logger
        self.streaming_service = StreamingService()

        # Initialize Mistral API configuration
        self.api_key = settings.mistral_api_key
        self.base_url = settings.mistral_base_url
        self.model = settings.mistral_model

        # Verify API key is available
        if not self.api_key:
            self.logger.warning("Mistral API key not found in environment variables")

        # Initialize agent state
        self.agent_initialized = False
        self.agents = {}

        # Initialize persona and prompt templates
        self.persona_config = {
            "role": "Expert Technical Instructor for the Physical AI & Humanoid Robotics Book",
            "tone": "authoritative but friendly, human's sounding instructor and mentor",
            "style": "technically precise",
            "constraints": [
                "never hallucinate",
                "never answer outside book content unless explicitly allowed",
                "clearly state uncertainty when context is insufficient"
            ]
        }

    async def initialize(self):
        """Initialize the Intelligence Service components."""
        try:
            # Verify API key availability
            if not self.api_key:
                raise ValueError("Mistral API key is required but not configured")

            # Disable tracing to avoid OPENAI_API_KEY requirement
            from agents import set_tracing_disabled
            set_tracing_disabled(True)

            # Initialize the main agent with LiteLLM for Mistral
            await self._initialize_main_agent()

            self.agent_initialized = True
            self.logger.info("Intelligence Service initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Intelligence Service: {str(e)}")
            raise

    async def _initialize_main_agent(self):
        """Initialize the main agent with tools and guardrails using LiteLLM for OpenRouter."""
        # Create specialized agents using LiteLLM model for OpenRouter API
        # Ensure model name is properly formatted for LiteLLM
        formatted_model = self.model
        if ':free' in self.model:
            # Handle OpenRouter free tier models by removing :free suffix
            formatted_model = self.model.replace(':free', '')

        # For Mistral, ensure the model is properly formatted
        # For Mistral models, prefix with 'mistral/' to help LiteLLM identify the provider
        if 'mistral' in self.base_url.lower():
            formatted_model = f"mistral/{formatted_model}"

        litellm_model = LitellmModel(
            model=formatted_model,  # Properly formatted for LiteLLM
            api_key=self.api_key,
            base_url=self.base_url
        )

        # Create model settings with usage tracking, temperature, and increased max turns
        model_settings = ModelSettings(temperature=0.8, include_usage=True, max_turns=20)

        self.agents["main"] = Agent(
            name="Technical Instructor Agent",
            instructions=f"""
            You are an {self.persona_config['role']}. Your persona characteristics are:
            - Tone: {self.persona_config['tone']}
            - Style: {self.persona_config['style']}
            - Constraints: {', '.join(self.persona_config['constraints'])}

            You must:
            - For greeting queries (hello, hi, hey, good morning, etc.), provide a friendly greeting and invite questions about the Physical AI & Humanoid Robotics book content
            - For specific technical questions, provide detailed answers based ONLY on the retrieved context
            - If the context doesn't contain relevant information for a specific query, acknowledge this limitation and suggest related topics from the context
            - Never speculate or fabricate information beyond what's in the provided context
            - Clearly state when context is insufficient to answer a specific question
            - Maintain technical precision and educational value
            - Use bullet points, numbered lists, and structured format when helpful
            - Always be specific and direct in your responses rather than generic
            - If a user asks why you're not giving proper answers, acknowledge the concern and explain that you can only answer based on the provided context
            - Focus on extracting and presenting the most relevant information from the context to directly address the user's query
            - Prioritize accuracy over completeness - better to acknowledge limitations than to provide incorrect information
            - When providing technical explanations, use clear language while maintaining precision
            - Structure responses to highlight the most important information first
            """,
            model=litellm_model,
            model_settings=model_settings,
            tools=[
                self.retrieve_book_context,  # Tool for retrieving book context
                self.load_conversation_context,  # Tool for loading conversation history
                self.validate_response_scope  # Tool for validating response scope
            ]
        )

    @function_tool
    async def retrieve_book_context(
        self,
        query: str,
        selected_text: Optional[str] = None
    ) -> str:  # Return JSON string to avoid schema issues
        """
        Retrieve book context based on the query and optional selected text.

        Args:
            query: User query to search for relevant context
            selected_text: Optional selected text from the book that may be relevant

        Returns:
            JSON string containing retrieved context chunks with metadata
        """
        import json
        from backend.services.retrieval_service import RetrievalService

        try:
            retrieval_service = RetrievalService()
            await retrieval_service.initialize()

            # Combine query with selected text if provided
            search_query = query
            if selected_text:
                search_query = f"{query} {selected_text}"

            # Validate query
            is_valid = await retrieval_service.validate_query(search_query)
            if not is_valid:
                result = {
                    "chunks": [],
                    "query": search_query,
                    "message": "Invalid query provided"
                }
                return json.dumps(result)

            # Perform retrieval using existing service (default top_k=5)
            sources = await retrieval_service.retrieve_by_query(
                query=search_query,
                top_k=5  # Default top-k as specified
            )

            # Format the results as a clean, ordered context block
            formatted_chunks = []
            for source in sources:
                chunk_data = {
                    "chunk_id": source.chunk_id,
                    "document_id": source.document_id,
                    "text": source.text,
                    "score": source.score,
                    "metadata": source.metadata
                }
                formatted_chunks.append(chunk_data)

            self.logger.info(f"Retrieved {len(formatted_chunks)} context chunks for query: {search_query[:50]}...")

            # Return JSON string
            result = {
                "chunks": formatted_chunks,  # Will be serialized by json.dumps
                "query": search_query,
                "retrieval_count": len(formatted_chunks),
                "message": "Successfully retrieved context chunks"
            }
            return json.dumps(result)

        except Exception as e:
            self.logger.error(f"Error in retrieve_book_context tool: {str(e)}")
            error_result = {
                "chunks": [],
                "query": query,
                "retrieval_count": 0,
                "message": f"Error retrieving context: {str(e)}"
            }
            return json.dumps(error_result)

    @function_tool
    async def load_conversation_context(
        self,
        session_id: str,
        max_turns: int = 5,
        max_tokens: int = 2000
    ) -> str:  # Return JSON string to avoid schema issues
        """
        Fetch recent conversation history from Neon database.

        Args:
            session_id: Unique identifier for the conversation session
            max_turns: Maximum number of conversation turns to retrieve
            max_tokens: Maximum number of tokens to include in context

        Returns:
            JSON string containing conversation history with proper formatting
        """
        import json
        try:
            # This would normally query the Neon database for conversation history
            # For now, we'll simulate with a basic structure
            # In a real implementation, this would connect to Neon and retrieve history

            # Placeholder for actual database query
            # conversation_history = await self.database_manager.get_conversation_history(
            #     session_id=session_id,
            #     max_turns=max_turns
            # )

            # Simulate conversation history
            conversation_history = {
                "session_id": session_id,
                "turns": [],
                "summary": "No previous conversation history found for this session.",
                "token_count": 0
            }

            # Format history safely
            formatted_history = {
                "conversation_context": conversation_history,
                "max_turns_used": max_turns,
                "max_tokens_allowed": max_tokens,
                "token_usage": conversation_history["token_count"],
                "has_context": len(conversation_history["turns"]) > 0
            }

            self.logger.info(f"Loaded conversation context for session: {session_id}")

            # Return JSON string
            result = {
                "conversation_context": conversation_history,
                "max_turns_used": max_turns,
                "max_tokens_allowed": max_tokens,
                "token_usage": conversation_history["token_count"],
                "has_context": len(conversation_history["turns"]) > 0
            }
            return json.dumps(result)

        except Exception as e:
            self.logger.error(f"Error in load_conversation_context tool: {str(e)}")
            error_result = {
                "conversation_context": {
                    "session_id": session_id,
                    "turns": [],
                    "summary": "Error retrieving conversation history",
                    "token_count": 0
                },
                "max_turns_used": max_turns,
                "max_tokens_allowed": max_tokens,
                "token_usage": 0,
                "has_context": False,
                "error": str(e)
            }
            return json.dumps(error_result)

    @function_tool
    async def validate_response_scope(
        self,
        response: str,
        context_chunks: str,  # JSON string representation of context chunks
        user_query: str
    ) -> str:  # Return JSON string to avoid schema issues
        """
        Validate that the response only references information from the provided context.

        Args:
            response: The generated response to validate
            context_chunks: JSON string representation of the context chunks that were provided to the agent
            user_query: The original user query

        Returns:
            JSON string containing validation results and safety assessment
        """
        import json

        try:
            # Parse the context chunks from JSON string
            parsed_chunks = []
            try:
                parsed_chunks = json.loads(context_chunks) if context_chunks else []
            except json.JSONDecodeError:
                # If it's not valid JSON, treat as raw string
                parsed_chunks = [{"text": context_chunks, "metadata": {}}]

            # Extract all text content from context chunks
            context_texts = []
            for chunk in parsed_chunks:
                if isinstance(chunk, dict):
                    text_content = chunk.get('text', '') or chunk.get('content', '')
                    context_texts.append(text_content)
                elif isinstance(chunk, str):
                    context_texts.append(chunk)

            combined_context = " ".join(context_texts)

            # Perform basic validation checks
            validation_results = {
                "response_length": len(response),
                "context_provided": len(parsed_chunks) > 0,
                "context_length": len(combined_context),
                "has_external_references": False,  # Will implement check
                "has_unsupported_claims": False,   # Will implement check
                "scope_compliance": True,          # Will implement check
                "hallucination_detected": False,   # Will implement check
                "confidence_score": 0.8           # Placeholder confidence
            }

            # Basic check: see if response content relates to context
            if len(combined_context) > 0:
                # Simple overlap check (in a full implementation, this would be more sophisticated)
                response_lower = response.lower()
                context_lower = combined_context.lower()

                # Count how much of the response seems to reference context topics
                context_words = set(context_lower.split())
                response_words = response_lower.split()

                matching_words = sum(1 for word in response_words if word in context_words)
                total_response_words = len(response_words)

                if total_response_words > 0:
                    context_alignment = matching_words / total_response_words
                    validation_results["context_alignment"] = context_alignment
                    validation_results["confidence_score"] = min(0.9, 0.5 + (context_alignment * 0.4))

                    # Flag potential issues
                    if context_alignment < 0.3:
                        validation_results["scope_compliance"] = False
                        validation_results["confidence_score"] *= 0.5

            # Check for common hallucination patterns
            hallucination_indicators = [
                "according to my training data",
                "i found online",
                "external source",
                "recent news",
                "latest research"  # Unless it's in the context
            ]

            response_lower = response.lower()
            for indicator in hallucination_indicators:
                if indicator in response_lower:
                    validation_results["hallucination_detected"] = True
                    validation_results["confidence_score"] *= 0.3
                    break

            # Final safety assessment
            validation_results["is_safe"] = (
                validation_results["scope_compliance"] and
                not validation_results["hallucination_detected"]
            )

            self.logger.info(f"Response validation completed for query: {user_query[:50]}...")

            # Return JSON string
            result = {
                "validation_results": validation_results,
                "original_response": response,
                "user_query": user_query,
                "passed_validation": validation_results["is_safe"],
                "message": "Response validated successfully" if validation_results["is_safe"] else "Response flagged for safety concerns"
            }
            return json.dumps(result)

        except Exception as e:
            self.logger.error(f"Error in validate_response_scope tool: {str(e)}")
            error_result = {
                "validation_results": {
                    "is_safe": False,
                    "confidence_score": 0.0,
                    "error": str(e)
                },
                "original_response": response,
                "user_query": user_query,
                "passed_validation": False,
                "message": f"Error during validation: {str(e)}"
            }
            import json
            return json.dumps(error_result)

    @input_guardrail
    async def query_input_guardrail(self, ctx, agent, input) -> GuardrailFunctionOutput:
        """
        Input guardrail to validate user queries before processing.
        """
        # Check if query is too short
        if len(input.strip()) < 3:
            return GuardrailFunctionOutput(
                output_info={"valid": False, "reason": "Query too short"},
                tripwire_triggered=True,
            )

        # Check if query contains inappropriate content
        inappropriate_keywords = ["joke", "opinion", "personal", "fictional", "make up", "imagine", "hypothetical"]
        input_lower = input.lower()
        has_inappropriate = any(keyword in input_lower for keyword in inappropriate_keywords)

        # Check if query asks for external information
        external_keywords = ["google", "search online", "recent news", "latest research", "external source"]
        has_external_request = any(keyword in input_lower for keyword in external_keywords)

        # Return validation results
        validation_result = not (has_inappropriate or has_external_request)

        return GuardrailFunctionOutput(
            output_info={
                "valid": validation_result,
                "has_inappropriate": has_inappropriate,
                "has_external_request": has_external_request
            },
            tripwire_triggered=not validation_result,
        )

    @output_guardrail
    async def response_output_guardrail(self, ctx, agent, input, output) -> GuardrailFunctionOutput:
        """
        Output guardrail to validate responses for safety and accuracy.
        """
        # Check if response contains hallucinated content
        hallucination_indicators = [
            "according to my training data",
            "i found online",
            "based on my knowledge",
            "from my training",
            "recent developments",
            "latest research",
            "external source",
            "recent news",
            "google",
            "search results",
            "i researched"
        ]

        has_hallucination = any(indicator in output.lower() for indicator in hallucination_indicators)

        # Check if response properly cites context
        context_related_indicators = ["provided context", "given information", "retrieved", "mentioned", "stated"]
        has_context_reference = any(indicator in output.lower() for indicator in context_related_indicators)

        # Check for refusal patterns (when context is insufficient)
        refusal_patterns = ["insufficient context", "cannot answer", "not enough information", "not in provided context"]
        has_proper_refusal = any(pattern in output.lower() for pattern in refusal_patterns)

        # Determine if response is safe
        is_safe = not has_hallucination or has_proper_refusal or has_context_reference

        return GuardrailFunctionOutput(
            output_info={
                "is_safe": is_safe,
                "has_hallucination": has_hallucination,
                "has_context_reference": has_context_reference,
                "has_proper_refusal": has_proper_refusal
            },
            tripwire_triggered=not is_safe,
        )

    async def process_query(
        self,
        user_query: str,
        context_chunks: List[Source],
        session_id: Optional[str] = None,
        persona_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Process a user query with retrieved context chunks using the Intelligence Layer.

        Args:
            user_query: The user's question that requires reasoning and response generation
            context_chunks: Pre-retrieved text chunks from Qdrant and PostgreSQL
            session_id: Optional session identifier for conversation memory
            persona_config: Optional persona configuration for response generation

        Returns:
            Dictionary containing the response and metadata
        """
        try:
            self.logger.info(f"Processing query: {user_query[:50]}... with {len(context_chunks)} context chunks")

            # Ensure agent is initialized before processing
            if not self.agent_initialized:
                self.logger.warning("Agent not initialized, attempting to initialize...")
                await self.initialize()

            # Validate inputs
            if not user_query or len(user_query.strip()) == 0:
                raise ValueError("User query cannot be empty")

            # Check if the query is a greeting - handle these specially
            user_query_lower = user_query.lower().strip()
            is_greeting = any(greeting in user_query_lower for greeting in ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'greetings', 'good evening'])

            if is_greeting:
                # For greetings, return a simple, appropriate response without using context
                greeting_response = "Hello! I'm your Physical AI & Humanoid Robotics book assistant. I can help answer questions about the content of the book. What would you like to know about Physical AI, Humanoid Robotics, ROS 2, or related topics?"

                final_response = {
                    "text": greeting_response,
                    "sources": [],
                    "structured_data": {},
                    "metadata": {
                        "processing_time": datetime.now().isoformat(),
                        "token_usage": {},
                        "confidence_score": 1.0,
                        "grounding_confirmed": True
                    }
                }
                return final_response

            # For non-greeting queries, proceed with normal processing
            if not context_chunks:
                self.logger.warning("No context chunks provided for query processing")

            # Prepare context for the agent
            context_text = "\n\n".join([chunk.text for chunk in context_chunks])

            # Create the main prompt using prompt engineering techniques
            prompt = self._create_layered_prompt(
                user_query=user_query,
                context_text=context_text,
                persona_config=persona_config or self.persona_config
            )

            # Create session if needed
            session = None
            if session_id:
                try:
                    session = SQLiteSession(session_id)
                    self.logger.debug(f"Created SQLite session for session_id: {session_id}")
                except Exception as session_error:
                    self.logger.error(f"Failed to create SQLite session: {str(session_error)}")
                    # Continue without session rather than failing completely
                    session = None

            # Initialize token_usage before the try block to avoid scope issues
            token_usage = {}

            # Run the agent
            try:
                self.logger.debug(f"Starting agent run for query: {user_query[:50]}...")
                result = await Runner.run(
                    self.agents["main"],
                    prompt,
                    session=session
                )

                response_text = result.final_output
                self.logger.debug(f"Agent run completed successfully, response length: {len(response_text)}")

                # Extract token usage if available
                if hasattr(result, 'context_wrapper') and hasattr(result.context_wrapper, 'usage'):
                    token_usage = result.context_wrapper.usage
                elif hasattr(result, 'usage'):
                    token_usage = result.usage
                else:
                    token_usage = {}

            except Exception as e:
                self.logger.error(f"Error running agent: {str(e)}")
                import traceback
                self.logger.error(f"Full traceback for agent run: {traceback.format_exc()}")
                # Fallback response
                response_text = f"I encountered an error while processing your query. Original query: {user_query}"

            # Validate response quality and grounding - use direct validation instead of tool call
            context_dicts = [{"text": chunk.text, "metadata": chunk.metadata, "score": chunk.score, "chunk_id": chunk.chunk_id, "document_id": chunk.document_id} for chunk in context_chunks]

            # Perform direct validation (not through the function tool since we can't call function tools directly)
            validated_response = await self._direct_validate_response_scope(
                response=response_text,
                context_chunks=context_dicts,
                user_query=user_query
            )

            # Add metadata to response
            final_response = {
                "text": validated_response.get("original_response", response_text),
                "sources": context_chunks,
                "structured_data": validated_response.get("structured_data", {}),
                "metadata": {
                    "processing_time": datetime.now().isoformat(),
                    "token_usage": token_usage,  # Token usage from LiteLLM
                    "confidence_score": validated_response.get("validation_results", {}).get("confidence_score", 0.0),
                    "grounding_confirmed": validated_response.get("validation_results", {}).get("is_safe", False)
                }
            }

            self.logger.info(f"Successfully processed query: {user_query[:50]}...")
            return final_response

        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return {
                "text": "An error occurred while processing your request. Please try again.",
                "sources": [],
                "structured_data": {},
                "metadata": {
                    "error": str(e),
                    "processing_time": datetime.now().isoformat()
                }
            }

    def _create_layered_prompt(
        self,
        user_query: str,
        context_text: str,
        persona_config: Dict
    ) -> str:
        """
        Create a layered prompt using prompt engineering techniques.

        This implements the CLEAR framework:
        - C: Context (the retrieved context)
        - L: Length (instructions for response length)
        - E: Example (would be examples of good responses)
        - A: Audience (the target audience)
        - R: Requirements (specific requirements for the response)
        """
        # Check if the query is a greeting
        user_query_lower = user_query.lower().strip()
        is_greeting = any(greeting in user_query_lower for greeting in ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'greetings'])

        # System instruction with role definition
        system_instruction = f"""
        SYSTEM INSTRUCTION:
        You are acting as a {persona_config['role']} with the following characteristics:
        - Tone: {persona_config['tone']}
        - Style: {persona_config['style']}
        - Constraints: {', '.join(persona_config['constraints'])}

        YOUR TASK:
        - For greeting queries (hi, hello, hey, etc.), provide a friendly greeting and invite questions about the Physical AI & Humanoid Robotics book content
        - For specific technical questions, answer based ONLY on the provided context below
        - Do NOT use any external knowledge or information from your training data
        - If the context doesn't contain sufficient information to answer the query, clearly state this and suggest related topics from the context
        - Maintain technical precision and educational value
        - Use bullet points, numbered lists, or structured format when helpful
        - Always cite relevant parts of the provided context when making claims
        - Focus on extracting and presenting the most relevant information to directly address the user's query
        """

        # Context section
        context_section = f"""
        RETRIEVED CONTEXT:
        {context_text}
        """

        # User query
        user_query_section = f"""
        USER QUERY:
        {user_query}
        """

        # Output requirements vary based on query type
        if is_greeting:
            output_requirements = """
            OUTPUT REQUIREMENTS FOR GREETINGS:
            - Respond with a friendly greeting
            - Introduce yourself as an expert on the Physical AI & Humanoid Robotics book
            - Invite the user to ask specific questions about the book content
            - Do NOT try to relate the greeting to the context content
            - Keep the response concise and welcoming
            """
        else:
            output_requirements = """
            OUTPUT REQUIREMENTS FOR TECHNICAL QUESTIONS:
            - Base your response primarily on the RETRIEVED CONTEXT provided above
            - Directly address the specific USER QUERY
            - If the context is insufficient, acknowledge this and suggest what related topics might be covered in the book
            - Format your response in a clear, educational manner with relevant details
            - Use bullet points or structured format when appropriate
            - Reference specific parts of the context when making claims
            - Be specific and direct rather than generic
            """

        # Combine all sections
        full_prompt = f"{system_instruction}\n\n{context_section}\n\n{user_query_section}\n\n{output_requirements}"

        return full_prompt

    async def stream_response(
        self,
        user_query: str,
        context_chunks: List[Source],
        session_id: Optional[str] = None,
        persona_config: Optional[Dict] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream response tokens incrementally to the client using OpenAI Agents SDK streaming.

        Args:
            user_query: The user's question
            context_chunks: Retrieved context chunks
            session_id: Optional session ID
            persona_config: Optional persona configuration

        Yields:
            Streamed response chunks in SSE format
        """
        try:
            # Ensure agent is initialized before processing
            if not self.agent_initialized:
                self.logger.warning("Agent not initialized, attempting to initialize...")
                await self.initialize()

            # Prepare context for the agent
            context_text = "\n\n".join([chunk.text for chunk in context_chunks])

            # Create the main prompt using prompt engineering techniques
            prompt = self._create_layered_prompt(
                user_query=user_query,
                context_text=context_text,
                persona_config=persona_config or self.persona_config
            )

            # Send context sources first
            for source in context_chunks:
                source_data = {
                    "type": "source",
                    "chunk_id": source.chunk_id,
                    "document_id": source.document_id,
                    "text": source.text[:200] + "..." if len(source.text) > 200 else source.text,
                    "score": source.score,
                    "metadata": source.metadata
                }
                yield f"data: {json.dumps(source_data)}\n\n"

            # Create session if needed
            session = None
            if session_id:
                try:
                    session = SQLiteSession(session_id)
                    self.logger.debug(f"Created SQLite session for session_id: {session_id}")
                except Exception as session_error:
                    self.logger.error(f"Failed to create SQLite session: {str(session_error)}")
                    # Continue without session rather than failing completely
                    session = None

            # Stream the agent response using the SDK's streaming capability
            try:
                self.logger.debug(f"Starting agent streaming for query: {user_query[:50]}...")

                # According to the OpenAI Agents SDK documentation, we should NOT await Runner.run_streamed()
                # It returns a RunResultStreaming object that provides stream_events()
                streaming_result = Runner.run_streamed(
                    self.agents["main"],
                    prompt,
                    session=session
                )

                # Track if we received any streaming events
                events_received = False

                # Access the async stream of StreamEvent objects for proper token-by-token streaming
                token_found = False
                self.logger.info("Starting to process streaming events from Agent SDK")
                try:
                    async for event in streaming_result.stream_events():
                        events_received = True

                        # Log event type for debugging
                        self.logger.info(f"Stream event type: {event.type}, event: {str(type(event))}, data type: {str(type(getattr(event, 'data', None)))}, has_data: {hasattr(event, 'data')}")

                        # Check if event has string representation that might contain text
                        event_str = str(event)
                        if 'text' in event_str.lower() or 'delta' in event_str.lower():
                            self.logger.info(f"Event string contains text/delta: {event_str[:200]}")

                        # FIRST: Handle specific OpenAI ResponseTextDeltaEvent which has the actual token content
                        # This is the most important event type for streaming text
                        if (hasattr(event, 'data') and
                            event.data and
                            type(event.data).__name__ == ResponseTextDeltaEvent and
                            hasattr(event.data, 'delta') and
                            event.data.delta):
                            delta_text = event.data.delta
                            if isinstance(delta_text, str) and delta_text.strip():
                                self.logger.info(f"Found ResponseTextDeltaEvent with delta: {repr(delta_text)[:100]}...")
                                chunk_data = {
                                    "type": "token",
                                    "content": delta_text,
                                }
                                yield f"data: {json.dumps(chunk_data)}\n\n"
                                token_found = True
                            continue  # Continue to next event after handling this specific type

                        # Handle various ResponseEvent types from OpenAI
                        if hasattr(event, 'data') and event.data:
                            # Safely check data type name to avoid exceptions
                            try:
                                data_type_name = type(event.data).__name__
                                # Handle ResponseTextDeltaEvent specifically - this is what contains the actual text deltas
                                if 'ResponseTextDeltaEvent' in data_type_name and hasattr(event.data, 'delta'):
                                    delta_text = event.data.delta
                                    if isinstance(delta_text, str) and delta_text.strip():
                                        self.logger.info(f"Processing {data_type_name} with delta: {repr(delta_text)[:100]}...")
                                        chunk_data = {
                                            "type": "token",
                                            "content": delta_text,
                                        }
                                        yield f"data: {json.dumps(chunk_data)}\n\n"
                                        token_found = True

                                # Handle other ResponseEvent types that might contain text content
                                elif hasattr(event.data, 'delta') and event.data.delta:
                                    delta_text = event.data.delta
                                    if isinstance(delta_text, str) and delta_text.strip():
                                        self.logger.debug(f"Found delta in event data: {repr(delta_text)[:100]}...")
                                        chunk_data = {
                                            "type": "token",
                                            "content": delta_text,
                                        }
                                        yield f"data: {json.dumps(chunk_data)}\n\n"
                                        token_found = True

                                elif hasattr(event.data, 'content') and event.data.content:
                                    content = event.data.content
                                    if isinstance(content, str) and content.strip():
                                        self.logger.debug(f"Found content in event data: {repr(content)[:100]}...")
                                        chunk_data = {
                                            "type": "token",
                                            "content": content,
                                        }
                                        yield f"data: {json.dumps(chunk_data)}\n\n"
                                        token_found = True

                                elif hasattr(event.data, 'text') and event.data.text:
                                    text_content = event.data.text
                                    if isinstance(text_content, str) and text_content.strip():
                                        self.logger.debug(f"Found text in event data: {repr(text_content)[:100]}...")
                                        chunk_data = {
                                            "type": "token",
                                            "content": text_content,
                                        }
                                        yield f"data: {json.dumps(chunk_data)}\n\n"
                                        token_found = True
                            except Exception as e:
                                self.logger.error(f"Error processing event data: {str(e)}")
                                continue  # Skip to next event if there's an error processing this one

                        # First, try to extract any possible text content directly from the event object
                        # Handle cases where the event itself might contain text data
                        try:
                            if hasattr(event, 'text') and event.text and isinstance(event.text, str) and event.text.strip():
                                content = event.text.strip()
                                self.logger.debug(f"Found text directly in event: {repr(content[:100])}...")
                                chunk_data = {
                                    "type": "token",
                                    "content": content,
                                }
                                yield f"data: {json.dumps(chunk_data)}\n\n"
                                token_found = True
                            elif hasattr(event, 'delta') and event.delta and isinstance(event.delta, str) and event.delta.strip():
                                content = event.delta.strip()
                                self.logger.debug(f"Found delta directly in event: {repr(content[:100])}...")
                                chunk_data = {
                                    "type": "token",
                                    "content": content,
                                }
                                yield f"data: {json.dumps(chunk_data)}\n\n"
                                token_found = True
                            elif hasattr(event, 'content') and event.content and isinstance(event.content, str) and event.content.strip():
                                content = event.content.strip()
                                self.logger.debug(f"Found content directly in event: {repr(content[:100])}...")
                                chunk_data = {
                                    "type": "token",
                                    "content": content,
                                }
                                yield f"data: {json.dumps(chunk_data)}\n\n"
                                token_found = True
                        except Exception as e:
                            self.logger.error(f"Error processing direct event attributes: {str(e)}")
                            continue  # Skip to next event if there's an error processing this one

                        # Handle different event types for proper streaming
                        # Focus on the most common event types from OpenAI Agents SDK
                        try:
                            if hasattr(event, 'type') and isinstance(event.type, str):
                                # Handle text delta events - common for streaming tokens
                                if "delta" in event.type.lower():
                                    if hasattr(event, 'data'):
                                        # Check for delta property in event data
                                        if hasattr(event.data, 'delta') and event.data.delta:
                                            delta_text = event.data.delta
                                            if isinstance(delta_text, str) and delta_text.strip():
                                                self.logger.debug(f"Yielding delta: {repr(delta_text)[:100]}...")
                                                chunk_data = {
                                                    "type": "token",
                                                    "content": delta_text,
                                                }
                                                yield f"data: {json.dumps(chunk_data)}\n\n"
                                                token_found = True  # Track that we found a token
                                        # Also check for content in event.data
                                        elif hasattr(event.data, 'content') and event.data.content:
                                            content = event.data.content
                                            if isinstance(content, str) and content.strip():
                                                self.logger.debug(f"Yielding content: {repr(content)[:100]}...")
                                                chunk_data = {
                                                    "type": "token",
                                                    "content": content,
                                                }
                                                yield f"data: {json.dumps(chunk_data)}\n\n"
                                                token_found = True  # Track that we found a token
                                        # Check text property in event data
                                        elif hasattr(event.data, 'text') and event.data.text:
                                            text_content = event.data.text
                                            if isinstance(text_content, str) and text_content.strip():
                                                self.logger.debug(f"Yielding text: {repr(text_content)[:100]}...")
                                                chunk_data = {
                                                    "type": "token",
                                                    "content": text_content,
                                                }
                                                yield f"data: {json.dumps(chunk_data)}\n\n"
                                                token_found = True  # Track that we found a token

                                # Handle message content events
                                elif "content" in event.type.lower() or "message" in event.type.lower():
                                    if hasattr(event, 'data') and hasattr(event.data, 'content'):
                                        content = event.data.content
                                        if isinstance(content, str) and content.strip():
                                            self.logger.debug(f"Yielding message content: {repr(content)[:100]}...")
                                            chunk_data = {
                                                "type": "token",
                                                "content": content,
                                            }
                                            yield f"data: {json.dumps(chunk_data)}\n\n"
                                            token_found = True  # Track that we found a token
                                    # Handle cases where event.data is directly a string
                                    elif isinstance(event.data, str) and event.data.strip():
                                        self.logger.debug(f"Yielding event data string: {repr(event.data)[:100]}...")
                                        chunk_data = {
                                            "type": "token",
                                            "content": event.data,
                                        }
                                        yield f"data: {json.dumps(chunk_data)}\n\n"
                                        token_found = True  # Track that we found a token

                                # Handle text-specific events
                                elif "text" in event.type.lower():
                                    if hasattr(event, 'data'):
                                        # Check for text in event data
                                        if hasattr(event.data, 'text') and event.data.text:
                                            text_content = event.data.text
                                            if isinstance(text_content, str) and text_content.strip():
                                                self.logger.debug(f"Yielding text event: {repr(text_content)[:100]}...")
                                                chunk_data = {
                                                    "type": "token",
                                                    "content": text_content,
                                                }
                                                yield f"data: {json.dumps(chunk_data)}\n\n"
                                                token_found = True  # Track that we found a token
                                        # Also check delta in text events
                                        elif hasattr(event.data, 'delta') and event.data.delta:
                                            delta_text = event.data.delta
                                            if isinstance(delta_text, str) and delta_text.strip():
                                                self.logger.debug(f"Yielding text delta: {repr(delta_text)[:100]}...")
                                                chunk_data = {
                                                    "type": "token",
                                                    "content": delta_text,
                                                }
                                                yield f"data: {json.dumps(chunk_data)}\n\n"
                                                token_found = True  # Track that we found a token

                                # Handle completion events
                                elif any(completion_word in event.type.lower() for completion_word in ['completed', 'done', 'finish', 'end']):
                                    self.logger.debug(f"Completion event received: {event.type}")
                                    break
                                else:
                                    # Log other event types for debugging
                                    self.logger.debug(f"Other event type processed: {event.type}")
                        except Exception as e:
                            self.logger.error(f"Error processing event type: {str(e)}")
                            continue  # Skip to next event if there's an error processing this one

                except Exception as e:
                    self.logger.error(f"Error in main streaming loop: {str(e)}")
                    # Send error message to client if we're within the streaming context
                    error_data = {
                        "type": "error",
                        "message": f"Streaming error: {str(e)}"
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    # Continue with fallback approach

                # If no streaming events were received, fall back to the non-streaming approach
                if not events_received:
                    self.logger.warning("No streaming events received, falling back to non-streaming approach")

                    # Use the regular process_query method and simulate streaming
                    result = await self.process_query(
                        user_query=user_query,
                        context_chunks=context_chunks,
                        session_id=session_id
                    )

                    response_text = result.get("text", "")
                    if response_text and response_text.strip():
                        # Send the response word by word to simulate true streaming
                        words = response_text.split()
                        for i, word in enumerate(words):
                            if word.strip():  # Only yield non-empty words
                                chunk_data = {
                                    "type": "token",
                                    "content": word + (" " if i < len(words) - 1 else ""),  # Add space except for last word
                                }
                                yield f"data: {json.dumps(chunk_data)}\n\n"
                    else:
                        # If no response text was generated, send a default message
                        chunk_data = {
                            "type": "token",
                            "content": "I couldn't find relevant information to answer your query.",
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"

                # Ensure that a completion message is always sent
                completion_data = {
                    "type": "complete",
                    "message": "Response completed"
                }
                yield f"data: {json.dumps(completion_data)}\n\n"

            except Exception as e:
                self.logger.error(f"Error in agent streaming: {str(e)}")
                import traceback
                self.logger.error(f"Full traceback: {traceback.format_exc()}")

                # Check if this is an API quota exceeded error by looking at the error message
                error_str = str(e).lower()
                error_str_full = str(e)  # Keep the full error string for more detailed checking
                friendly_message = "An error occurred while processing your request. Please try again."

                # Check for various API quota/exceedance related phrases
                if any(phrase in error_str for phrase in ['quota', 'credit', 'limit', 'exceeded', 'rate limit', 'usage limit', 'api key', 'payment required']) or \
                   any(phrase in error_str_full for phrase in ['credits', 'more credits', 'afford', 'paid account', 'upgrade']):
                    friendly_message = "We've reached our API usage limit. Please try again later or check back soon!"
                elif any(phrase in error_str for phrase in ['connection', 'timeout', 'network', 'connectivity']):
                    friendly_message = "We're having trouble connecting to our services. Please check your internet connection and try again."
                elif any(phrase in error_str for phrase in ['authentication', 'auth', '401', '403', 'unauthorized']):
                    friendly_message = "We're having trouble accessing our services. Please try again later."

                # Send error message to client
                error_data = {
                    "type": "error",
                    "message": friendly_message
                }
                yield f"data: {json.dumps(error_data)}\n\n"

                # Send completion message even in error case
                completion_data = {
                    "type": "complete",
                    "message": "Response completed"
                }
                yield f"data: {json.dumps(completion_data)}\n\n"

        except Exception as e:
            self.logger.error(f"Error in streaming response: {str(e)}")
            error_str = str(e).lower()
            error_str_full = str(e)  # Full error string for more detailed checking

            # Provide friendly error messages for common issues
            if any(phrase in error_str for phrase in ['quota', 'credit', 'limit', 'exceeded', 'rate limit', 'usage limit', 'api key', 'payment required']) or \
               any(phrase in error_str_full.lower() for phrase in ['more credits', 'afford', 'paid account', 'upgrade', 'max_tokens']):
                friendly_message = "We've reached our API usage limit. Please try again later or check back soon!"
            elif any(phrase in error_str for phrase in ['connection', 'timeout', 'network', 'connectivity']):
                friendly_message = "We're having trouble connecting to our services. Please check your internet connection and try again."
            elif any(phrase in error_str for phrase in ['authentication', 'auth', '401', '403', 'unauthorized']):
                friendly_message = "We're having trouble accessing our services. Please try again later."
            else:
                friendly_message = "An error occurred while processing your request. Please try again."

            error_data = {
                "type": "error",
                "message": friendly_message
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    async def validate_query_and_context(
        self,
        user_query: str,
        context_chunks: List[Source]
    ) -> bool:
        """
        Validate query and context before processing.

        Args:
            user_query: The user's query
            context_chunks: Retrieved context chunks

        Returns:
            True if valid, False otherwise
        """
        if not user_query or len(user_query.strip()) == 0:
            self.logger.warning("Query validation failed: empty query")
            return False

        if len(user_query.strip()) < 3:
            self.logger.warning("Query validation failed: query too short")
            return False

        # Additional validation can be added here

        return True

    async def _direct_validate_response_scope(
        self,
        response: str,
        context_chunks: List[Dict[str, Any]],  # Simplified type
        user_query: str
    ) -> Dict[str, Any]:  # Simplified return type to avoid schema issues
        """
        Direct validation of response quality and grounding without using function tools.
        This is used internally instead of calling the function tool directly.
        """
        try:
            # Extract all text content from context chunks
            context_texts = []
            for chunk in context_chunks:
                if isinstance(chunk, dict):
                    context_texts.append(chunk.get('text', ''))
                elif hasattr(chunk, 'text'):
                    context_texts.append(chunk.text)

            combined_context = " ".join(context_texts)

            # Perform basic validation checks
            validation_results = {
                "response_length": len(response),
                "context_provided": len(context_chunks) > 0,
                "context_length": len(combined_context),
                "has_external_references": False,  # Will implement check
                "has_unsupported_claims": False,   # Will implement check
                "scope_compliance": True,          # Will implement check
                "hallucination_detected": False,   # Will implement check
                "confidence_score": 0.8           # Placeholder confidence
            }

            # Basic check: see if response content relates to context
            if len(combined_context) > 0:
                # Simple overlap check (in a full implementation, this would be more sophisticated)
                response_lower = response.lower()
                context_lower = combined_context.lower()

                # Count how much of the response seems to reference context topics
                context_words = set(context_lower.split())
                response_words = response_lower.split()

                matching_words = sum(1 for word in response_words if word in context_words)
                total_response_words = len(response_words)

                if total_response_words > 0:
                    context_alignment = matching_words / total_response_words
                    validation_results["context_alignment"] = context_alignment
                    validation_results["confidence_score"] = min(0.9, 0.5 + (context_alignment * 0.4))

                    # Flag potential issues
                    if context_alignment < 0.3:
                        validation_results["scope_compliance"] = False
                        validation_results["confidence_score"] *= 0.5

            # Check for common hallucination patterns
            hallucination_indicators = [
                "according to my training data",
                "i found online",
                "external source",
                "recent news",
                "latest research"  # Unless it's in the context
            ]

            response_lower = response.lower()
            for indicator in hallucination_indicators:
                if indicator in response_lower:
                    validation_results["hallucination_detected"] = True
                    validation_results["confidence_score"] *= 0.3
                    break

            # Final safety assessment
            validation_results["is_safe"] = (
                validation_results["scope_compliance"] and
                not validation_results["hallucination_detected"]
            )

            self.logger.info(f"Response validation completed for query: {user_query[:50]}...")

            # Convert complex types to simple types to avoid schema issues
            return {
                "validation_results": validation_results,
                "original_response": response,
                "user_query": user_query,
                "passed_validation": validation_results["is_safe"],
                "message": "Response validated successfully" if validation_results["is_safe"] else "Response flagged for safety concerns"
            }

        except Exception as e:
            self.logger.error(f"Error in direct validate_response_scope: {str(e)}")
            return {
                "validation_results": {
                    "is_safe": False,
                    "confidence_score": 0.0,
                    "error": str(e)
                },
                "original_response": response,
                "user_query": user_query,
                "passed_validation": False,
                "message": f"Error during validation: {str(e)}"
            }