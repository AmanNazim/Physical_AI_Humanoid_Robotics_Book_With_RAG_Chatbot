# Constitution: RAG Chatbot for Physical AI Humanoid Robotics Book

## Abstract

This constitution defines the highest-level architecture, philosophy, constraints, rules, system guarantees, and component boundaries for the RAG (Retrieval-Augmented Generation) Chatbot system designed for the Physical AI Humanoid Robotics Book. This system provides users with accurate answers to questions based strictly on book content through intelligent retrieval and reasoning capabilities, using UV as the package manager and Chatkit for the user interface.

## Objectives

- Enable users to ask questions about the Physical AI Humanoid Robotics Book and receive accurate answers based solely on book content
- Support both whole-book retrieval and user-selected text-only retrieval modes
- Provide fast, scalable, and explainable AI-powered responses with proper source citations
- Maintain deterministic retrieval to prevent generative hallucination
- Ensure cloud-free-tier friendly operation for cost-effectiveness
- Use UV as the package manager for dependency management
- Implement user interface using Chatkit framework

## System Philosophy

### Design Principles

1. **Deterministic retrieval over generative hallucination**: The system must prioritize retrieving and citing actual book content over generating responses from internal knowledge or imagination.

2. **Ground truth from book text**: All responses must be grounded in and derived from the Physical AI Humanoid Robotics Book content, never from external sources or internal model knowledge.

3. **Clean, normalized embeddings and metadata**: All text chunks must be processed with consistent tokenization and normalization to ensure reliable retrieval.

4. **Latency optimization**: The system must leverage FastAPI's async capabilities and Qdrant's caching to deliver responses within performance constraints.

5. **Explainable reasoning path**: The system must provide clear reasoning paths and source citations for all responses.

6. **Modularity for future models**: Architecture must allow for future model replacements or enhancements without major rewrites.

## Core Modules

### 1. Embedding Pipeline

- **Chunking**: Text processing module that splits book content into manageable chunks (800-1200 tokens)
- **Embedding generation**: Uses Cohere API to generate vector embeddings for each text chunk
- **Vector uploading**: Uploads generated embeddings to Qdrant vector database
- **Metadata saving**: Stores chunk metadata to Neon Postgres database
- **Storage Layer**: Manages the interface between embedding generation and storage systems

### 2. Storage Layer

- **Qdrant Vector DB**: Primary vector storage for embeddings with similarity search capabilities
- **Neon Postgres**: Serverless Postgres for metadata, logs, chat history, and relational data
- **Data schemas**: Well-defined schemas for chunk metadata, user interactions, and system logs
- **Indexing requirements**: Proper indexing strategies for efficient retrieval and metadata queries

### 3. FastAPI Backend

- **API endpoints**: RESTful endpoints for chat interactions, document management, and system operations
- **Retrieval pipeline**: Coordinates the retrieval process from query to response
- **Async processing**: Implements asynchronous processing for improved performance and scalability
- **Agent routing endpoints**: Routes requests to the intelligence layer for processing
- **Authentication**: Optional authentication layer for user management

### 4. UI Layer (Chatkit)

- **User interface**: Implements the chat interface using Chatkit framework
- **Chat display**: Renders conversation history and responses
- **User input**: Handles user queries and mode selection
- **Response presentation**: Displays answers with source citations

### 5. Intelligence Layer (OpenAI Agents SDK)

- **Reasoning**: Performs intelligent reasoning based on retrieved context
- **Retrieval orchestration**: Coordinates the retrieval process and manages context
- **Tool usage**: Integrates retriever and postgres tools for data access
- **Context-constrained answering**: Ensures answers are only generated from allowed context
- **User-selected text mode**: Handles specialized mode where only user-selected text is used

## Constraints

- **Max retrieval latency**: Response time must be less than 1.5 seconds
- **Max embedding chunk size**: Text chunks must be between 800-1200 tokens
- **Embedding model**: Must use Cohere's free API key for embedding generation
- **Storage systems**: Limited to Qdrant for vectors and Neon Postgres for metadata
- **Cloud tier compatibility**: All components must remain compatible with free tiers
- **Framework compatibility**: Must maintain 100% compatibility with FastAPI and OpenAI Agents SDK
- **Package manager**: Must use UV for dependency management
- **UI framework**: Must implement the user interface using Chatkit

## Rules

- **No imagination-based answers**: The system must never generate answers from internal knowledge or imagination; all responses must be based on retrieved book content.
- **Source citation requirement**: The agent must always cite the source chunks that inform each response.
- **Selected text mode enforcement**: When user enables "selected text only" mode, only those specific chunks may be used for response generation.
- **Consistent embedding parameters**: Embeddings must be generated with consistent tokenizer and chunk size settings across all processing.
- **Qdrant exclusivity**: Qdrant is the only approved vector storage system; no other vector stores may be introduced.
- **Neon Postgres exclusivity**: Neon Postgres is the only approved relational storage system.
- **No additional storage systems**: No other storage systems may be introduced without constitutional amendment.
- **Free tier compliance**: All components must remain cloud-free-tier friendly to maintain cost-effectiveness.

## Data Governance

### Required Metadata Fields

- Chunk ID: Unique identifier for each text chunk
- Source document: Reference to the original document/chapter
- Page/chapter reference: Location within the book
- Chunk text: The actual text content
- Embedding vector: Vector representation for similarity search
- Creation timestamp: When the chunk was processed
- Processing version: Version of the embedding pipeline used

### Privacy Requirements

- No user personal data may be embedded in the vector database
- Logs stored in Postgres must exclude private user information
- User chat history should be anonymized where possible
- All personal information must be handled in compliance with privacy regulations

## Interfaces

### Embedding Pipeline → Qdrant Interface

- **Vector insertion**: Vectors are inserted with consistent dimensionality and metadata
- **Metadata storage**: Chunk metadata is stored as payload in Qdrant with proper indexing
- **Batch processing**: Supports efficient batch uploads of multiple vectors at once

### FastAPI → Qdrant + Postgres Interface

- **Retrieval workflow**: Query transformation, vector search, result ranking, and metadata enrichment
- **Query transformation**: Converts user queries to appropriate vector search parameters
- **Result aggregation**: Combines vector search results with metadata from Postgres

### FastAPI → Agent Interface

- **Agent receives user question**: The original query from the user
- **Agent receives relevant chunks**: Retrieved text chunks relevant to the query
- **Agent receives metadata**: Additional metadata about retrieved chunks
- **Agent receives mode settings**: Information about whether to use "whole book" or "selected text only" mode

### Agent SDK → LLM Interface

- **Retriever tool binding**: The agent must bind the retriever tool for context access
- **Postgres tool binding**: The agent must bind the postgres tool for metadata access
- **Context requirement**: The agent must not call the LLM without proper context from book content

## Future Scalability Notes

- The modular architecture allows for easy replacement of embedding models or vector databases
- The FastAPI backend supports horizontal scaling through standard deployment patterns
- The system can accommodate additional books or documents with minimal changes
- The agent-based architecture allows for complex reasoning workflows as requirements evolve
- The cloud-native design enables easy migration to paid tiers as usage grows

## Final Declaration

This constitution serves as the foundational document governing the RAG Chatbot for Physical AI Humanoid Robotics Book. All future specifications, plans, and tasks must align with the principles, constraints, and architectural decisions outlined herein. Any significant deviations require a constitutional amendment process to ensure continued alignment with the system's core purpose and design philosophy.