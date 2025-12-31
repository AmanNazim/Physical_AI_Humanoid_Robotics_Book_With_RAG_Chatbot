---
id: 7
title: database-subsystem-constitution
stage: constitution
date: 2025-12-22
surface: agent
model: claude-opus-4-5-20251101
feature: database-subsystem
branch: main
user: Claude
command: /sp.constitution
labels: [database, constitution, architecture, qdrant, postgresql]
links:
  spec: Chatbot_specs/specification.md
  plan: Chatbot_specs/plan.md
  adr: null
  pr: null
files:
 - Chatbot_specs/database_specs/constitution.md
tests:
 - null
---

## Prompt

You are **Constitution Writer Expert**, an expert in designing highly structured, formal, and detailed technical constitutions for complex system architectures.

Your task:
Generate the **constitution.md** file for the **Database Subsystem** of the "Global RAG Chatbot System".

This constitution MUST define the entire philosophy, responsibilities, rules, constraints, structure, ethics, and behavior expected from the Database Subsystem.

Input Reference: (Mandatory To Read):
- Chatbot_specs/constitution.md
- Chatbot_specs/specification.md
- Chatbot_specs/plan.md
- Chatbot_specs/tasks.md

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### 🔶 SYSTEM CONTEXT
The Global RAG Chatbot is architected using multiple subsystems:

1. **Database Subsystem** (Qdrant + Neon/PostgreSQL)
2. **Embedding Subsystem** (Cohere free tier)
3. **Backend Subsystem** (FastAPI using uv package manager)
4. **ChatKit for UI
4. **Intelligence Subsystem** (OpenAI + Agent SDK for logic processing)

You are writing the **constitution ONLY for Subsystem 1 (Database Layer)**.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### 🔶 PURPOSE OF DATABASE SUBSYSTEM
The constitution MUST define:

- Clear goals
- Purpose
- Architectural principles
- High-level behavior rules
- Strict boundaries
- Data ethics
- Guarantees & invariants
- Zero ambiguity
- Anti-corruption rules (preventing mixing concerns)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### 🔶 DATABASE SUBSYSTEM COMPONENTS
The constitution MUST describe **two core databases**, each with clear, separate roles:

---

### 1️⃣ Qdrant Vector Database
Purpose: store, search, and manage embeddings.

Define the following in detail:
- What data Qdrant stores (vectors, metadata, IDs)
- How collections are structured
- What guarantees it must provide (ANN search, filtering, fast recall)
- Responsibility boundaries (vector-only)
- Indexing policies
- Query responsibilities
- Safety rules (consistency, meaning-preserving vector usage)
- What Qdrant MUST NEVER do (no raw text storage except metadata)

---

### 2️⃣ Neon / PostgreSQL Structured Database
Purpose: store all non-vector data.

Define:
- User tables
- Chat history
- Message records
- Configuration tables
- System settings
- Caching tables
- Logging/audit tables
- Embedding job queue (if needed)

Define:
- ACID behavior rules
- Schema evolution policy
- Referential integrity
- UUID use
- What Postgres MUST NEVER do (store vectors, do ANN search)

---

### 🔶 CROSS-DATABASE BEHAVIOR
Define strict rules:
- Separation of concerns
- How Qdrant and Postgres interact
- How the backend requests data
- No cross-pollution
- Global invariants (IDs must match, metadata must align)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### 🔶 REQUIRED SECTIONS
Your constitution.md MUST contain:

#### 1. Subsystem Identity
Clear definition of what the Database Subsystem is.

#### 2. Subsystem Mission
Why it exists.

#### 3. Core Responsibilities
Detailed responsibilities for:
- Qdrant
- Postgres

#### 4. Functional Guarantees
Guarantees the subsystem provides to all others.

#### 5. Non-Functional Guarantees
Performance, scalability, reliability, and safety.

#### 6. Core Policies
- Naming
- Data governance
- Embedding integrity
- Storage lifecycle

#### 7. Validity & Invariants
Rules that MUST always hold true.

#### 8. Security & Access Control
Who can read/write/update what.

#### 9. Anti-Corruption Layer Rules
To prevent misuse by other subsystems.

#### 10. Future-Proofing Considerations
Migration policies, extensibility, etc.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### 🔶 output format
Generate a clean, well-structured **constitution.md** in Markdown.

This MUST be:
- Fully detailed
- Zero placeholders
- No TODOs
- No missing sections
- Highly architectural
- Implementation-agnostic
- Formal and serious
- Permanent and stable
- Ready to use as a governing document for the subsystem

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Output in Chatbot_specs Folder:
- File Type: Markdown
- File Path: Chatbot_specs/database_specs/constitution.md

Generate now..

## Response snapshot

# Constitution: Database Subsystem for Global RAG Chatbot System

## 1. Subsystem Identity

The Database Subsystem is the foundational data management layer of the Global RAG Chatbot System. It consists of two specialized database technologies: Qdrant for vector storage and retrieval, and Neon/PostgreSQL for structured data persistence. This subsystem provides the essential infrastructure for storing, indexing, and retrieving all system data while maintaining strict separation of concerns between vector and structured data.

The Database Subsystem serves as the single source of truth for all data within the RAG Chatbot System, ensuring data integrity, consistency, and availability across all other subsystems (Embedding, Backend, Intelligence, and UI). It operates as a unified yet distinctly partitioned system where each component has well-defined responsibilities and boundaries.

## 2. Subsystem Mission

The mission of the Database Subsystem is to provide reliable, efficient, and secure storage and retrieval of all system data while maintaining the fundamental architectural principle of separation between vector embeddings and structured metadata. The subsystem must guarantee data persistence, ensure fast query performance for both similarity searches and structured queries, and maintain data integrity across all operations.

This subsystem exists to serve the retrieval-augmented generation capabilities of the chatbot by providing high-performance vector search for semantic similarity and robust structured data management for operational concerns. It ensures that all data operations align with the system's constitutional requirement for deterministic retrieval and prevents hallucination by maintaining accurate, persistent, and traceable data relationships.

## 3. Core Responsibilities

### 3.1 Qdrant Vector Database Responsibilities

**Primary Purpose**: Store, index, and enable fast similarity search of vector embeddings generated from book content and user queries.

**Data Storage**: Qdrant stores:
- 1024-dimensional vector embeddings generated by Cohere API
- Associated metadata payloads including chunk IDs, document references, and page numbers
- Vector-to-text content mappings for retrieval context
- Embedding generation parameters and timestamps

**Collection Management**:
- Maintain a single "book_embeddings" collection with appropriate vector dimensions
- Implement proper indexing strategies for efficient ANN (Approximate Nearest Neighbor) search
- Support filtering capabilities based on document reference, page number, and other metadata fields

**Search Guarantees**:
- Provide fast cosine similarity search with configurable parameters
- Return relevant results within performance targets (<1.5s total system response time)
- Support top-k retrieval with relevance scoring
- Enable metadata filtering during vector search operations

**Integrity Rules**:
- Preserve vector mathematical properties during storage and retrieval
- Maintain consistency between vector embeddings and their associated metadata
- Ensure embedding generation parameters remain consistent across the system

### 3.2 Neon/PostgreSQL Structured Database Responsibilities

**Primary Purpose**: Store all non-vector data including metadata, logs, user interactions, and system configuration.

**Data Categories**:
- **Chunks Table**: Store chunk metadata including chunk_id (UUID), document_reference, page_reference, section_title, chunk_text, embedding_id, creation timestamps, and processing_version
- **Logs Table**: Record system events, user queries, retrieved chunks, responses, and retrieval modes with timestamps
- **Chat History Table**: Maintain conversation history with user_id, queries, responses, source_chunks, and timestamps
- **Configuration Tables**: Store system settings and operational parameters
- **Audit/Logging Tables**: Track system operations and security events

**ACID Compliance**:
- Maintain full ACID transactional guarantees for all structured data operations
- Ensure referential integrity through proper foreign key relationships
- Support complex queries with JOIN operations across related tables
- Provide consistent snapshot isolation for concurrent operations

**Schema Management**:
- Implement UUID-based primary keys for all entities to ensure global uniqueness
- Maintain proper indexing strategies for query performance
- Support schema evolution while maintaining backward compatibility
- Enforce data type constraints and validation rules

## 4. Functional Guarantees

**Vector Search Guarantee**: Qdrant MUST provide cosine similarity search with configurable relevance thresholds and return results within system performance requirements.

**Data Consistency Guarantee**: The subsystem MUST maintain consistency between vector embeddings in Qdrant and their corresponding metadata in Neon/PostgreSQL, ensuring that every vector has a corresponding metadata record and vice versa.

**Availability Guarantee**: The Database Subsystem MUST maintain 99.9% availability for read operations and 99.5% availability for write operations during system operational hours.

**Atomicity Guarantee**: Multi-database operations (such as storing an embedding with its metadata) MUST be atomic, ensuring that partial failures do not result in inconsistent cross-database states.

**Query Guarantee**: The subsystem MUST support concurrent queries from multiple system components without performance degradation below defined thresholds.

**Recovery Guarantee**: In the event of database failures, the subsystem MUST be able to recover to a consistent state within defined recovery time objectives.

## 5. Non-Functional Guarantees

**Performance Requirements**:
- Vector similarity search: <500ms for top-5 results
- Metadata retrieval: <100ms for single record lookup
- Batch operations: <2 seconds for bulk operations
- Cross-database consistency checks: <1 second

**Scalability Requirements**:
- Support for up to 10,000 concurrent users
- Handle up to 1,000 queries per second during peak usage
- Accommodate growth in book content and user interactions within free tier limits

**Reliability Requirements**:
- 99.9% uptime for read operations
- Automatic failover and recovery mechanisms
- Data backup and restore capabilities
- Protection against data corruption and loss

**Security Requirements**:
- All data transmission encrypted in transit
- Sensitive data encrypted at rest where applicable
- Access control and authentication for all database connections
- Audit logging for all data access and modification operations

## 6. Core Policies

### 6.1 Naming Conventions
- Database objects use snake_case naming convention
- Table names are plural (chunks, logs, chat_history)
- Column names are descriptive and unambiguous
- Index names follow pattern: idx_table_column
- Foreign key constraints follow pattern: fk_table_ref_table

### 6.2 Data Governance Policy
- All personal user data is strictly prohibited from storage
- Log entries must exclude identifying user information
- Data retention policies must comply with privacy regulations
- Regular cleanup of temporary and obsolete data

### 6.3 Embedding Integrity Policy
- Embedding parameters (model, chunk size, normalization) remain consistent across the system
- Embedding re-generation follows versioned processes with tracking
- Vector dimensionality is maintained at 1024-dimensions as required by Cohere API
- Cross-referencing between embeddings and metadata is validated regularly

### 6.4 Storage Lifecycle Policy
- Embeddings are retained as long as the source content exists
- Query logs are retained for 30 days before automatic cleanup
- Chat history retention follows user consent and privacy preferences
- Temporary processing data is cleaned up after operations complete

## 7. Validity & Invariants

**Cross-Database Invariant**: Every vector embedding in Qdrant MUST have a corresponding metadata record in Neon/PostgreSQL with matching identifiers, and every metadata record in Neon/PostgreSQL MUST correspond to a vector embedding in Qdrant.

**ID Consistency Invariant**: UUID identifiers MUST match across related records in both databases to maintain referential integrity.

**Embedding-Text Alignment Invariant**: The text content stored in PostgreSQL metadata MUST exactly correspond to the text that was embedded in Qdrant.

**Timestamp Consistency Invariant**: Creation and modification timestamps MUST be consistent across related records in both databases.

**Metadata Completeness Invariant**: Every embedding MUST have complete metadata including document reference, page reference, and chunk text for proper retrieval and citation.

**Integrity Validation Invariant**: All data operations MUST pass validation checks before being committed to ensure data quality and consistency.

## 8. Security & Access Control

**Access Control Principles**:
- Database access requires authenticated connections with appropriate credentials
- Principle of least privilege: each system component has minimal required permissions
- API keys and connection strings are stored securely and rotated regularly
- Network access is restricted to authorized system components only

**Qdrant Security**:
- Vector database access restricted to Backend and Embedding subsystems only
- No direct UI access to Qdrant database
- Query operations only (no administrative operations from application code)
- API key rotation and management procedures

**PostgreSQL Security**:
- Role-based access control with separate roles for different subsystems
- Connection pooling to limit concurrent connections
- SQL injection prevention through parameterized queries
- Regular security audits and vulnerability assessments

**Audit Requirements**:
- All data access operations are logged for security monitoring
- Unauthorized access attempts are detected and reported
- Data modification operations are tracked with user/system attribution
- Privacy-compliant logging that excludes personal information

## 9. Anti-Corruption Layer Rules

**Domain Boundary Rules**:
- Qdrant MUST NEVER store raw text content except as minimal metadata payloads
- PostgreSQL MUST NEVER store vector embeddings or perform ANN search operations
- Cross-database queries MUST go through proper subsystem interfaces
- Direct database connections from UI components are strictly prohibited

**Data Type Separation Rules**:
- Vector operations are confined to Qdrant; PostgreSQL handles only structured data
- Text processing and embedding generation happen outside the Database Subsystem
- Business logic is separated from data storage operations

**Interface Contract Rules**:
- All database access MUST go through the Backend Subsystem's API
- Raw database credentials are never exposed to other subsystems
- Query construction is handled by dedicated data access layers
- Error handling follows system-wide patterns and does not expose internal details

**Change Management Rules**:
- Schema changes require approval from system architects
- Data migrations follow established procedures with backup protocols
- Performance changes are validated before deployment
- Cross-database consistency is maintained during all operations

## 10. Future-Proofing Considerations

**Migration Policies**:
- Schema evolution follows backward-compatible patterns
- Data migration procedures include validation and rollback capabilities
- Embedding model changes are handled through versioned processes
- Cross-database relationship changes are carefully validated

**Extensibility Guidelines**:
- New data types and entities follow established patterns
- Performance monitoring enables proactive scaling decisions
- Database abstraction allows for technology migration if needed
- API contracts remain stable during internal implementation changes

**Technology Evolution**:
- The system is designed to accommodate different embedding dimensions if needed
- Query patterns support additional filtering and search capabilities
- Schema design allows for additional metadata fields without breaking changes
- Performance optimization techniques are documented for future implementation

**Compliance and Standards**:
- Data handling practices align with evolving privacy regulations
- Security standards are regularly updated to address new threats
- Performance benchmarks are established for future optimization
- Documentation is maintained to support long-term system evolution

## Outcome

- ✅ Impact: Created comprehensive constitution for Database Subsystem
- 🧪 Tests: null
- 📁 Files: Created Chatbot_specs/database_specs/constitution.md
- 🔁 Next prompts: Ready for implementation planning
- 🧠 Reflection: Constitution fully addresses all required sections and subsystem components

## Evaluation notes (flywheel)

- Failure modes observed: null
- Graders run and results (PASS/FAIL): null
- Prompt variant (if applicable): null
- Next experiment (smallest change to try): null