# Specification: RAG Chatbot for "Physical AI Humanoid Robotics" Book

## 1. System Overview

### Purpose of the RAG Chatbot
The RAG (Retrieval-Augmented Generation) Chatbot for "Physical AI Humanoid Robotics" Book is designed to answer user questions strictly based on book content. The system leverages advanced retrieval mechanisms to provide accurate, contextually-relevant responses while preventing hallucination by grounding all answers in the book's actual text.

### Workflow Summary
The system follows a multi-stage workflow:
1. Book content is processed and embedded into vector representations
2. User submits a question via the ChatKit UI
3. FastAPI backend receives the query and determines retrieval mode
4. Vector search retrieves relevant text chunks from Qdrant
5. OpenAI Agents SDK orchestrates the reasoning process
6. Response is generated based on retrieved context and delivered via ChatKit UI

### Five Major Subsystems
1. **Embedding Pipeline (Cohere Free Tier)**: Handles text chunking, normalization, and embedding generation
2. **Storage Layer (Qdrant Cloud + Neon Serverless Postgres)**: Manages vector storage and metadata persistence
3. **FastAPI Backend (managed with `uv` package manager)**: Provides API endpoints and orchestration
4. **UI Layer (Chatkit)**: Implements the user interface and user interaction handling
5. **Intelligence Layer (OpenAI Agents SDK)**: Performs reasoning and answer generation

### ChatKit Frontend Integration
The system includes a ChatKit-based frontend that provides a user-friendly interface for interacting with the RAG system, including mode selection (full-book vs. selected-text-only) and source citation display.

### Retrieval Modes
- **Full-book retrieval**: Answers based on the entire book content
- **User-selected text only**: Answers exclusively from user-provided text snippets

## 2. Functional Requirements

### Core Functionality
- **Question Processing**: User submits question → FastAPI receives → retrieval → Agent → answer
- **Mode Selection**: Support for both full-book and selected-text-only modes
- **Source Citation**: Chatbot must cite exact book chunks used in answers
- **Hallucination Prevention**: Chatbot must never generate answers beyond provided text context
- **Text Preprocessing**: Clean normalization, chunking, and preprocessing of input text

### User Interaction Flow
1. User enters question in ChatKit interface
2. FastAPI receives query and optional mode selection
3. System retrieves relevant context using vector search
4. Agent generates answer based on retrieved context
5. Response with source citations is displayed to user

## 3. Technical Requirements

### 3.1 Embeddings (Cohere Free Tier)
- **Chunk Size**: 800–1200 tokens
- **Input Normalization**: Text is cleaned of extraneous whitespace and special characters
- **Batch Embedding Strategy**: Process multiple text chunks in parallel to optimize API usage
- **Embedding Retry Logic**: Implement exponential backoff for failed embedding requests
- **Embedding Caching**: Cache results to avoid redundant embedding of identical content

### 3.2 Storage Layer

#### Qdrant Cloud Free Tier
- **Collection Name**: `book_embeddings`
- **Vector Schema**: 1024-dimensional vectors (matching Cohere embedding dimensions)
- **Payload Schema**:
  - `chunk_id`: Unique identifier for the text chunk
  - `text_content`: The actual text content of the chunk
  - `document_reference`: Reference to the source document/chapter
  - `metadata`: Additional metadata including page numbers, section titles
- **Index Type**: Auto-index on metadata fields for efficient filtering
- **Distance Metric**: Cosine distance for similarity search
- **Query Filters**: Support for filtering by document reference and metadata

#### Neon Serverless Postgres
- **Metadata Tables**:
  - `chunks`: Stores chunk metadata and relationships
  - `logs`: Records query logs and system events
  - `chat_history`: Maintains conversation history (optional)

**Table Schemas:**

**chunks table:**
```
chunk_id (UUID, primary key)
document_reference (VARCHAR) -- Source document: Reference to the original document/chapter
page_reference (INTEGER) -- Page/chapter reference: Location within the book
section_title (VARCHAR)
chunk_text (TEXT) -- Chunk text: The actual text content
embedding_id (UUID)
created_at (TIMESTAMP) -- Creation timestamp: When the chunk was processed
updated_at (TIMESTAMP)
processing_version (VARCHAR) -- Processing version: Version of the embedding pipeline used
```

**logs table:**
```
log_id (UUID, primary key)
user_query (TEXT)
retrieved_chunks (JSONB)
response (TEXT)
timestamp (TIMESTAMP)
retrieval_mode (VARCHAR) -- 'full_book' or 'selected_text'
```

**chat_history table:**
```
chat_id (UUID, primary key)
user_id (UUID)
query (TEXT)
response (TEXT)
source_chunks (JSONB)
timestamp (TIMESTAMP)
```

- **Required Indexes**: Indexes on document_reference, created_at, and chunk_id
- **Foreign Keys**: Foreign key relationships between related tables
- **Size Limitations**: Adhere to Neon's free tier storage limits

## 4. Backend Specifications — FastAPI (using uv)

### Project Folder Structure
```
rag-chatbot/
├── src/
│   ├── main.py
│   ├── config/
│   │   └── settings.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── query.py
│   │   │   ├── embed.py
│   │   │   ├── retrieve.py
│   │   │   └── selected_text.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── request_models.py
│   │   └── response_models.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── embedding_service.py
│   │   ├── retrieval_service.py
│   │   └── agent_service.py
│   └── utils/
│       ├── __init__.py
│       └── text_processor.py
├── tests/
├── pyproject.toml
├── uv.lock
└── README.md
```

### Required Python Version
- Python 3.10 or higher

### Required uv Commands
- `uv sync`: Install dependencies from pyproject.toml
- `uv run`: Execute Python scripts with project dependencies
- `uv lock`: Generate/update uv.lock file

### Async API Architecture
- All endpoints implemented as async functions
- Use asyncio for concurrent processing
- Background tasks for batch operations

### API Endpoints

#### `/query` (POST)
- Accepts user questions and mode selection
- Returns answers with source citations
- Request body: `{ "question": "user question", "mode": "full_book|selected_text_only" }`
- Response: `{ "answer": "generated answer", "source_chunks": ["chunk_id1", "chunk_id2"] }`

#### `/embed` (POST)
- Processes and embeds text chunks
- Request body: `{ "text_chunks": ["text1", "text2"] }`
- Response: `{ "status": "success", "chunk_ids": ["id1", "id2"] }`

#### `/retrieve` (POST)
- Performs vector search and retrieves relevant chunks
- Request body: `{ "query": "search query", "top_k": 5 }`
- Response: `{ "chunks": [{"id": "chunk_id", "text": "chunk_text", "score": 0.9}] }`

#### `/selected-text` (POST)
- Processes user-provided text for selected-text-only mode
- Request body: `{ "text": "user provided text", "question": "user question" }`
- Response: `{ "answer": "generated answer", "source_chunks": ["provided_text"] }`

#### `/agent/route` (POST)
- Routes requests to the OpenAI Agent for processing
- Request body: `{ "user_input": "user query", "context": "retrieved context" }`
- Response: `{ "response": "agent response", "citations": ["chunk_id1", "chunk_id2"] }`

#### `/health` (GET)
- Returns system health status
- Response: `{ "status": "healthy", "timestamp": "ISO timestamp" }`

### Background Tasks
- Batch embedding processing for large document sets
- Periodic cleanup of temporary data
- Asynchronous logging of interactions

### Rate Limiting
- Implement rate limiting to comply with Cohere API usage limits
- Limit to 100 requests per minute per IP

### Response Format Standards
- Consistent JSON responses across all endpoints
- Standardized error responses with error codes and messages

## 5. Agent Intelligence Layer (OpenAI Agents SDK)

### Tools
- **Vector Search Tool**: Interfaces with Qdrant for context retrieval
- **Metadata/Postgres Tool**: Interfaces with Neon Postgres for metadata lookups
- **Citation Tool**: Generates source citations for responses

### Context Assembly Pipeline
1. Receive user query and mode from FastAPI
2. Retrieve relevant chunks based on mode
3. Assemble context from retrieved chunks
4. Pass context to agent for response generation

### Rules for Preventing Ungrounded LLM Answers
- Agent must not generate responses without proper context
- If no relevant chunks are retrieved, respond with "I cannot find information about this in the book"
- Agent must always reference source chunks in responses

### Agent Behavior Requirements
- **Use Retrieved Chunks**: Agent must incorporate only retrieved text in responses
- **Respect Constitution Rules**: Follow all constitutional constraints on behavior
- **Avoid Hallucination**: Never generate information not present in retrieved context
- **Cite Chunk IDs**: Include references to source chunks in all responses

## 6. ChatKit UI Integration

### UI Request Flow
1. User types question in chat interface
2. ChatKit sends request to FastAPI `/query` endpoint
3. Response is displayed in chat window with source citations
4. Loading indicators show processing status

### Components
- **Chat Box**: Main interface for user questions and bot responses
- **Document Selection Widget**: For "selected text only" mode
- **Loading Indicators**: Show when system is processing
- **Source Chunk Viewer**: Displays cited sources for transparency

### ChatKit-FastAPI Communication
- REST API calls from ChatKit to FastAPI endpoints
- WebSocket connections for real-time streaming if implemented
- CORS configured for secure communication

### Real-time Streaming
- Optional real-time response streaming for improved user experience
- Support for streaming responses from the OpenAI Agent

## 7. Data Flow Specifications

### Ingestion Pipeline
```
Book → Text Chunker → Cohere Embedding → Qdrant Vector Storage → Neon Postgres Metadata
```

1. Book content is parsed and split into chunks of 800-1200 tokens
2. Each chunk is sent to Cohere for embedding generation
3. Generated embeddings are stored in Qdrant with metadata
4. Metadata is stored in Neon Postgres for reference and tracking

### Query Pipeline
```
User Question → FastAPI → Qdrant Similarity Search → Agent Processing → LLM Answer → ChatKit UI
```

1. User submits question via ChatKit UI
2. FastAPI receives and processes the query
3. Vector similarity search performed in Qdrant
4. Retrieved chunks passed to OpenAI Agent
5. Agent generates answer using retrieved context
6. Response with citations returned to ChatKit UI

### Selected Text Mode Pipeline
```
User-Supplied Text → Embed On-the-Fly → Agent Processing → Direct Answer → ChatKit UI
```

1. User provides specific text and question
2. Text is embedded in real-time
3. Agent processes using only provided text
4. Response is generated without additional retrieval
5. Answer is returned to UI

## 8. Non-Functional Requirements

### Performance Requirements
- **Latency Target**: Response time must be less than 1.5 seconds (<1.5s)
- **Throughput**: Support for up to 10 concurrent users
- **Caching**: Implement caching for frequently accessed chunks

### Cost Minimization
- Optimize API calls to stay within free tier limits
- Implement intelligent caching to reduce redundant operations
- Use serverless architecture to minimize infrastructure costs
- All components must remain cloud-free-tier friendly to maintain cost-effectiveness

### Free Tier Compliance
- Adhere to Qdrant Cloud Free Tier limits
- Stay within Neon Serverless Postgres storage limits
- Comply with Cohere API usage limits

### Reliability Requirements
- Implement retry logic for external API calls
- Provide graceful degradation when services are unavailable
- Maintain system availability during peak usage

## 9. Security & Authentication

### Optional BetterAuth Integration
- Support for optional user authentication
- Session management for persistent conversations
- User data isolation

### CORS Requirements
- Configure CORS to allow requests from trusted origins only
- Implement proper security headers

### Token Validation
- Validate API tokens for Cohere, Qdrant, and Neon
- Secure token storage using environment variables

### Data Privacy
- No user personal data may be embedded in the vector database
- Logs stored in Postgres must exclude private user information
- No personal user data may be stored or embedded
- All user queries are logged without identifying information
- User chat history should be anonymized where possible
- All personal information must be handled in compliance with privacy regulations
- Conversation history is optional and user-controlled

## 10. Deployment Specification

### Backend Deployment
- Deploy to Railway, Render, or Fly.io
- Use environment variables for configuration
- Implement health checks and monitoring

### Required Environment Variables
- `COHERE_API_KEY`: API key for Cohere embeddings
- `QDRANT_API_KEY`: API key for Qdrant Cloud
- `NEON_DATABASE_URL`: Connection string for Neon Postgres
- `OPENAI_API_KEY`: API key for OpenAI Agents SDK
- `UV_ENVIRONMENT`: Configuration for uv package manager

### UV Build Configuration
- Use `uv sync` to install dependencies in deployment
- Include uv.lock in version control for reproducible builds
- Optimize build times by caching dependencies

### Credentials Loading
- Securely load all API credentials from environment variables
- Implement fallback mechanisms for credential loading
- Validate credentials at startup

## 11. Acceptance Criteria

### Core Functionality
- [ ] System can answer any question grounded in book content
- [ ] In selected-text-only mode, answers exclusively use provided text snippets
- [ ] Vector search retrieves relevant passages with high accuracy
- [ ] System responds with "I cannot find information" when no relevant content exists
- [ ] All 4 subsystems operate independently and are modularly replaceable

### Performance
- [ ] Response time consistently under 1.5 seconds
- [ ] System handles 10 concurrent users without degradation
- [ ] Vector search returns results within 500ms

### Reliability
- [ ] System provides graceful error handling
- [ ] Failed API calls are retried appropriately
- [ ] System operates within free tier limits consistently

### Quality Assurance
- [ ] No hallucination occurs in any response
- [ ] All answers include proper source citations
- [ ] Selected text mode exclusively uses provided content
- [ ] System maintains constitutional constraints
- [ ] System operates within cloud-free-tier limits consistently