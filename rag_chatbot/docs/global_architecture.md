# Global Architecture Documentation - RAG Chatbot System

## Overview

This document describes the global architecture of the RAG (Retrieval-Augmented Generation) Chatbot system for the Physical AI Humanoid Robotics Book. The system is designed to answer user questions based strictly on book content, preventing hallucination by grounding all responses in the provided context.

## System Architecture

### High-Level Flow

```
User Query → FastAPI Backend → RAG Pipeline → [Retrieval → Agent Processing] → Response → ChatKit UI
```

### Subsystem Boundaries

#### 1. Backend (FastAPI)
- **Responsibility**: API endpoints, request handling, orchestration
- **Location**: `/backend`
- **Key Components**:
  - Main application (`main.py`)
  - Route handlers (`routes/`)
  - API contracts

#### 2. Frontend (ChatKit)
- **Responsibility**: User interface, interaction handling
- **Location**: `/frontend`
- **Key Components**:
  - Chat interface
  - Mode selection
  - Response display

#### 3. Agents SDK Layer
- **Responsibility**: Intelligence layer, reasoning, answer generation
- **Location**: `/agents_sdk`
- **Key Components**:
  - Agent interfaces
  - Reasoning pipeline
  - Tool integration

#### 4. Databases Layer
- **Responsibility**: Data storage, metadata management
- **Location**: `/databases`
- **Key Components**:
  - Qdrant vector storage
  - Neon Postgres metadata
  - Connection utilities

#### 5. Embedding Pipeline
- **Responsibility**: Text processing, chunking, embedding generation
- **Location**: `/embedding_pipeline`
- **Key Components**:
  - Text chunker
  - Embedding service
  - Storage integration

#### 6. RAG Core
- **Responsibility**: Shared retrieval logic, utilities, interfaces
- **Location**: `/rag_core`
- **Key Components**:
  - RAG pipeline
  - Interface contracts
  - Utility functions

#### 7. Shared Components
- **Responsibility**: Configuration, schemas, constants
- **Location**: `/shared`
- **Key Components**:
  - Configuration management
  - Pydantic schemas
  - Shared constants

## Integration Rules

### Interface Contracts
- All subsystems must implement the defined interfaces in `/rag_core/interfaces`
- Interface contracts ensure loose coupling between components
- Subsystems can be replaced without affecting others if contracts are maintained

### Data Flow
- Text flows from ingestion → embedding → vector storage → retrieval → response
- Metadata flows through Neon Postgres database
- Configuration is centralized in `/shared/config.py`

### Error Handling
- All subsystems must use the error schemas defined in `/shared/schemas/error.py`
- Errors should be propagated with proper error codes and messages
- Graceful degradation is expected when services are unavailable

## Folder Responsibilities

### `/backend`
- FastAPI application entry point
- API route definitions
- Request/response validation
- Health checks

### `/frontend`
- ChatKit UI components
- User interaction handling
- API communication
- Response presentation

### `/agents_sdk`
- OpenAI Agent integration
- Reasoning pipeline
- Tool definitions
- Context assembly

### `/databases`
- Qdrant client utilities
- Neon Postgres utilities
- Connection pooling
- Migration scripts

### `/embedding_pipeline`
- Text preprocessing
- Chunking logic
- Embedding generation
- Batch processing

### `/rag_core`
- Core RAG pipeline logic
- Interface definitions
- Shared utilities
- Common constants

### `/shared`
- Configuration management
- Pydantic schemas
- Constants and enums
- Common utilities

### `/scripts`
- Admin scripts
- Data migration
- Setup utilities
- Maintenance tools

### `/docs`
- Architecture documentation
- API documentation
- Setup guides
- Troubleshooting

## Key Integration Points

### Configuration
- All services read from the central configuration in `/shared/config.py`
- Environment variables are loaded using pydantic-settings
- Settings are validated at startup

### Interfaces
- Subsystems communicate through defined interfaces in `/rag_core/interfaces`
- Each subsystem implements the relevant interface contracts
- This allows for modular replacement of components

### Schemas
- Request/response validation uses Pydantic models defined in `/shared/schemas`
- All API endpoints use the same schema definitions
- This ensures consistency across the system

## Security Considerations

- API keys are loaded from environment variables only
- No credentials are hardcoded in the source code
- CORS is configured through settings
- Input validation is performed using Pydantic schemas