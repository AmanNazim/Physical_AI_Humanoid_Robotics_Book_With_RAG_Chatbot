from fastapi import APIRouter
from ...services.query_service import QueryService
from ...services.rag_service import RAGService
from ...models.request_models import QueryRequest, ChatRequest
from ...models.response_models import QueryResponse, ChatResponse, SearchResponse
from ...models.response_models import Source

router = APIRouter(tags=["query"])


@router.post("/search")
async def search_endpoint(query_request: QueryRequest) -> SearchResponse:
    """
    Vector search wrapper endpoint.

    Args:
        query_request: QueryRequest containing the search query

    Returns:
        SearchResponse with search results
    """
    query_service = QueryService()
    await query_service.initialize()

    # Retrieve context for the query (which is essentially searching)
    sources = await query_service.retrieve_context(
        query_request.query,
        query_request.max_context
    )

    return SearchResponse(
        results=sources,
        query=query_request.query
    )


@router.post("/semantic-search")
async def semantic_search_endpoint(query_request: QueryRequest) -> SearchResponse:
    """
    Semantic search endpoint.

    Args:
        query_request: QueryRequest containing the search query

    Returns:
        SearchResponse with search results
    """
    query_service = QueryService()
    await query_service.initialize()

    # Retrieve context for the query (which is essentially searching)
    sources = await query_service.retrieve_context(
        query_request.query,
        query_request.max_context
    )

    return SearchResponse(
        results=sources,
        query=query_request.query
    )


@router.post("/hybrid-search")
async def hybrid_search_endpoint(query_request: QueryRequest) -> SearchResponse:
    """
    Hybrid search endpoint (Neon + Qdrant).

    Args:
        query_request: QueryRequest containing the search query

    Returns:
        SearchResponse with search results
    """
    query_service = QueryService()
    await query_service.initialize()

    # For now, we'll use the same approach as semantic search
    # In a full implementation, this would combine keyword and vector search
    sources = await query_service.retrieve_context(
        query_request.query,
        query_request.max_context
    )

    return SearchResponse(
        results=sources,
        query=query_request.query
    )


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest) -> ChatResponse:
    """
    Main RAG endpoint for chat functionality.

    Args:
        chat_request: ChatRequest containing the user query and session info

    Returns:
        ChatResponse with the answer and sources
    """
    query_service = QueryService()
    await query_service.initialize()
    return await query_service.process_chat(chat_request)


@router.post("/conversation-state")
async def conversation_state_endpoint():
    """
    Endpoint for managing conversation state.

    Returns:
        Response with conversation state management
    """
    # Placeholder for conversation state management
    return {"status": "success", "message": "Conversation state endpoint - to be implemented"}