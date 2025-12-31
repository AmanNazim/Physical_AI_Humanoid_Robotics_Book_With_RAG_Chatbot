---
id: 8
title: database-subsystem-specification
stage: spec
date: 2025-12-22
surface: agent
model: claude-opus-4-5-20251101
feature: database-subsystem
branch: main
user: Claude
command: /sp.specify
labels: [database, specification, architecture, qdrant, postgresql]
links:
  spec: Chatbot_specs/specification.md
  plan: Chatbot_specs/plan.md
  constitution: Chatbot_specs/database_specs/constitution.md
  adr: null
  pr: null
files:
 - Chatbot_specs/database_specs/specification.md
tests:
 - null
---

## Prompt

You are **Specification Architect Expert**, an expert in designing highly detailed, deeply structured technical specifications for modular software systems.

Your task:
Generate the **specification.md** file for the **Database Subsystem** of the "Global RAG Chatbot System".

This specification defines **WHAT** the subsystem must do (requirements), not HOW it will be implemented.
The specification MUST be strict, measurable, complete, and implementation-agnostic.

Input Reference: (Mandatory To Read):
- Chatbot_specs/constitution.md
- Chatbot_specs/specification.md
- Chatbot_specs/plan.md
- Chatbot_specs/tasks.md
- Chatbot_specs/database_specs/constitution.md


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### 🔶 SYSTEM CONTEXT
The Global RAG Chatbot System contains five subsystems:

1. Database Subsystem (Qdrant + Neon/PostgreSQL)
2. Embedding Subsystem (Cohere free tier)
3. Backend Subsystem (FastAPI using uv)
4. Intelligence Subsystem (Agent SDK)
5. ChatKit UI

You are creating the specification ONLY for subsystem 1.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### 🔶 REQUIRED OUTPUT STRUCTURE
Your **specification.md** MUST contain the following sections with complete detail:

---

## 1. Subsystem Overview
- Summary of the role of the Database Subsystem
- Purpose in the context of RAG
- Boundary definitions (what is inside/outside)

---

## 2. High-Level Requirements
Define all top-level requirements, such as:

### 2.1 Functional Requirements
- Storage of embeddings
- Storage of chat messages
- Storage of users
- Storage of chunked documents
- Retrieval requirements
- Vector similarity search requirements
- Search performance constraints
- Metadata handling
- Query filtering capabilities
- Message history retrieval rules
- Indexing requirements
- Versioning requirements
- Data lifecycle and cleanup

### 2.2 Non-Functional Requirements
- Latency targets
- Throughput targets
- Scaling expectations
- Availability expectations
- Data consistency requirements
- Fault tolerance
- Durability
- Migration constraints
- Auditability requirements
- Backups

---

## 3. Detailed Role of Each Database

### 3.1 Qdrant Vector Database
Define Qdrant's responsibilities in measurable detail:
- Vector storage
- Embedding metadata schema
- Collection naming rules
- Vector ID requirements
- ANN search guarantees
- Filtering support
- Payload structure and constraints
- Vector update rules
- Delete/replace guarantees
- Required indexes
- Query requirements for RAG workflows
- Integrity constraints
- What Qdrant may NOT store or do

### 3.2 Neon / PostgreSQL
Define PostgreSQL responsibilities:
- User accounts schema requirements
- Chat history schema requirements
- Document schema requirements
- Settings/config tables
- Logging/audit table structure
- Embedding job queue (optional)
- Foreign-key policies
- Indexing requirements
- Data normalization level
- Concurrency rules
- Transaction requirements
- Prohibited behaviors (e.g., no vector storage)

---

## 4. Cross-Database Interaction Specifications
Define measurable requirements for how the two databases coordinate:
- Shared IDs
- Metadata synchronization
- Backend request contracts
- Cross-database consistency guarantees
- Error-handling expectations
- Retry strategies
- No duplication of responsibilities allowed

---

## 5. API Interface Requirements
Define how backend communicates with the Database Subsystem:
- CRUD operations
- Vector search API
- Query filtering options
- Pagination requirements
- Batch operations
- Error formats
- Timeout requirements
- Required responses
- Required metadata fields

---

## 6. Data Models
Define all model specifications:

- User model
- Chat message model
- Document model
- Chunk model
- Vector model
- Settings model
- Audit log model

For each model define:
- Required fields
- Optional fields
- Data types
- Constraints
- Validation rules
- Relationships
- ID format rules
- Versioning requirements
- Soft/hard delete behavior

(NO actual schema or SQL — this is specification-only)

---

## 7. Security Requirements
Specify database security rules:
- Encryption expectations
- Access control per table/collection
- Role-based permissions
- Allowed request origins
- Input validation assumptions
- Rate-limiting expectations
- Secure query patterns
- Compliance requirements
- Logging & auditing expectations
- Secrets handling expectations

---

## 8. Operational Requirements
Define:
- Deployment expectations
- Performance monitoring
- Health checks
- Backup frequency
- Alerting rules
- Maintenance window policies
- Migration guidelines

---

## 9. Anti-Corruption / Boundary Rules
This section MUST prohibit misuse:
- Backend cannot bypass the Database Subsystem
- Intelligence subsystem cannot modify the database
- Embedding subsystem cannot read user data
- Qdrant cannot be used for relational queries
- Postgres cannot be used as vector index
- No direct inter-database writes
- No circular dependencies

---

## 10. Future Extensions
Define optional but expected future capabilities:
- Multi-vector embeddings
- Semantic cache
- Vector re-ranking database
- Additional metadata fields
- Multi-tenant support
- Hybrid search
- Sharding or distributed architecture

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### 🔶 OUTPUT REQUIREMENTS
- MUST be full, long, complete, and extremely detailed
- MUST NOT include vague statements or TODOs
- MUST follow strict specification language
- MUST be implementation-agnostic
- MUST fully align with constitution.md
- MUST be written as a real engineering-level spec document

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Output in Chatbot_specs Folder:
- File Type: Markdown
- File Path: Chatbot_specs/database_specs/specification.md

Generate the **specification.md** now..

## Response snapshot

# Specification: Database Subsystem for Global RAG Chatbot System

## 1. Subsystem Overview

The Database Subsystem serves as the foundational data management layer of the Global RAG Chatbot System. It consists of two specialized database technologies: Qdrant for vector storage and retrieval, and Neon/PostgreSQL for structured data persistence. This subsystem provides the essential infrastructure for storing, indexing, and retrieving all system data while maintaining strict separation of concerns between vector and structured data.

The purpose of this subsystem in the context of RAG (Retrieval-Augmented Generation) is to enable fast, accurate retrieval of relevant information from the Physical AI Humanoid Robotics Book content. The subsystem ensures that vector embeddings accurately represent the source content while maintaining proper metadata relationships that allow the intelligence layer to ground responses in the original text.

**Boundary Definitions**:
- **Inside**: Vector storage and retrieval, structured data persistence, metadata management, cross-database consistency, query interfaces, security controls
- **Outside**: Business logic, application-level processing, user interface management, embedding generation, AI reasoning

## 2. High-Level Requirements

### 2.1 Functional Requirements

**Storage of Embeddings**:
- The subsystem MUST store 1024-dimensional vector embeddings generated by Cohere API
- Each embedding MUST be associated with its original text content and metadata
- Embeddings MUST be stored with a unique identifier that matches corresponding structured data records
- The system MUST support bulk storage operations for large document sets

**Storage of Chat Messages**:
- The subsystem MUST store all chat messages with associated metadata including user_id, timestamp, and source_chunks
- Each chat session MUST be identifiable and retrievable by user context
- Message content MUST be preserved exactly as generated or received
- The system MUST maintain conversation context across multiple exchanges

**Storage of Users**:
- User accounts MUST be stored with unique identifiers, creation timestamps, and optional metadata
- Authentication-related data MUST be stored securely with appropriate hashing
- User preferences and settings MUST be preserved persistently
- Privacy-compliant user data handling MUST be enforced

**Storage of Chunked Documents**:
- Document content MUST be stored with complete metadata including source reference, page numbers, and section titles
- Chunk boundaries MUST be preserved to maintain context for retrieval
- Text content MUST be stored verbatim without modification
- Processing version information MUST be maintained for traceability

**Retrieval Requirements**:
- The system MUST support retrieval of data based on semantic similarity
- Exact match queries MUST be supported for structured data
- Partial match queries MUST be supported with configurable thresholds
- Complex filtering MUST be supported across metadata fields

**Vector Similarity Search Requirements**:
- Cosine similarity search MUST be supported with configurable thresholds
- Top-k retrieval MUST be supported with configurable result counts
- Metadata filtering MUST be supported during similarity search
- Query expansion capabilities MUST be available for enhanced recall

**Search Performance Constraints**:
- Vector similarity searches MUST return results within 500ms for top-5 retrieval
- Structured queries MUST return results within 100ms for single record lookups
- Complex queries with multiple filters MUST complete within 1 second
- Batch operations MUST complete within 2 seconds for standard batch sizes

**Metadata Handling**:
- All metadata MUST be stored with proper data types and constraints
- Metadata relationships MUST be preserved across database boundaries
- Versioning information MUST be maintained for all metadata updates
- Metadata integrity MUST be validated during all write operations

**Query Filtering Capabilities**:
- The system MUST support filtering by document reference
- Page and section-based filtering MUST be available
- Timestamp-based range queries MUST be supported
- Content-based filtering MUST be available for metadata fields

**Message History Retrieval Rules**:
- Chat history MUST be retrievable by user session
- Message ordering MUST be preserved chronologically
- Associated metadata (source chunks, citations) MUST be included with retrieved messages
- History retention policies MUST be configurable and enforced

**Indexing Requirements**:
- Vector indexes MUST be created for efficient similarity search
- Metadata field indexes MUST be created for fast filtering
- Cross-reference indexes MUST be maintained for joint queries
- Index maintenance MUST not impact query performance during operation

**Versioning Requirements**:
- Embedding version information MUST be tracked and stored
- Document versioning MUST be maintained for content evolution
- Schema versioning MUST be tracked for data model changes
- Processing pipeline versioning MUST be maintained for traceability

**Data Lifecycle and Cleanup**:
- Temporary processing data MUST be automatically cleaned up after operations
- Query logs MUST be retained for 30 days before automatic deletion
- Expired user sessions MUST be cleaned up according to retention policies
- Inactive chat history MAY be archived according to configurable policies

### 2.2 Non-Functional Requirements

**Latency Targets**:
- Vector similarity search: <500ms for top-5 results
- Metadata retrieval: <100ms for single record lookup
- Batch operations: <2 seconds for standard batch sizes
- Cross-database consistency checks: <1 second

**Throughput Targets**:
- Support for up to 1,000 queries per second during peak usage
- Handle up to 10,000 concurrent users without performance degradation
- Process bulk embedding operations at 100+ chunks per minute
- Support 99th percentile response times within specified targets

**Scaling Expectations**:
- Horizontal scaling capabilities MUST be available for future growth
- Vertical scaling MUST be supported for increased load
- Storage capacity MUST scale with book content growth
- Performance MUST remain consistent with increasing data volumes

**Availability Expectations**:
- 99.9% uptime for read operations during system operational hours
- 99.5% uptime for write operations during system operational hours
- Failover capabilities MUST be available for both database systems
- Recovery time objectives MUST be met during system failures

**Data Consistency Requirements**:
- Cross-database consistency MUST be maintained for all related records
- ACID properties MUST be preserved for structured data operations
- Vector-metadata alignment MUST be guaranteed across all operations
- Consistency checks MUST be performed regularly and automatically

**Fault Tolerance**:
- The system MUST continue operating with single database failures
- Automatic retry mechanisms MUST be implemented for transient failures
- Circuit breaker patterns MUST be available for dependent service failures
- Graceful degradation MUST be implemented during high load conditions

**Durability**:
- All data MUST be persisted durably across system restarts
- Redundant storage mechanisms MUST be implemented for critical data
- Backup capabilities MUST be available for disaster recovery
- Data integrity checks MUST be performed regularly

**Migration Constraints**:
- Schema migrations MUST be backward compatible
- Data migrations MUST maintain cross-database consistency
- Migration processes MUST include validation and rollback capabilities
- Zero-downtime migration strategies MUST be employed where possible

**Auditability Requirements**:
- All data access operations MUST be logged with appropriate detail
- User data access MUST be auditable for compliance purposes
- System operations MUST be traceable for debugging and analysis
- Audit logs MUST be maintained for compliance retention periods

**Backups**:
- Regular automated backups MUST be performed for all data
- Backup retention policies MUST be configurable and enforced
- Backup integrity verification MUST be performed regularly
- Point-in-time recovery capabilities MUST be available

## 3. Detailed Role of Each Database

### 3.1 Qdrant Vector Database

**Vector Storage**:
- Qdrant MUST store 1024-dimensional vector embeddings generated by Cohere API
- Each vector MUST be stored with proper dimensionality configuration
- Vector storage MUST support efficient compression techniques where applicable
- Storage format MUST preserve vector mathematical properties

**Embedding Metadata Schema**:
- Payload MUST include chunk_id as a searchable field
- Payload MUST include document_reference for content filtering
- Payload MUST include page_reference for location tracking
- Payload MAY include section_title for context preservation
- Payload MAY include processing_version for pipeline tracking

**Collection Naming Rules**:
- Collections MUST use the name "book_embeddings" for primary vector storage
- Collection naming MUST follow standard naming conventions
- Additional collections MAY be created for specific use cases with proper naming prefixes
- Collection naming MUST not conflict with system reserved names

**Vector ID Requirements**:
- Vector IDs MUST correspond to matching structured data record IDs
- ID format MUST be UUID-compliant for global uniqueness
- IDs MUST remain consistent across system operations and restarts
- ID assignment MUST be deterministic based on content for reproducibility

**ANN Search Guarantees**:
- Approximate nearest neighbor search MUST provide configurable accuracy vs speed trade-offs
- Search results MUST maintain consistent ranking across multiple queries
- Search performance MUST meet specified latency requirements
- Search recall MUST maintain minimum acceptable thresholds

**Filtering Support**:
- Metadata filtering MUST be supported during vector search operations
- Range-based filtering MUST be supported for numeric metadata fields
- Text-based filtering MUST be supported for string metadata fields
- Boolean filtering MUST be supported for flag-type metadata fields

**Payload Structure and Constraints**:
- Payload size MUST not exceed Qdrant's maximum payload limits
- Payload structure MUST remain consistent across all vector records
- Required payload fields MUST be validated during write operations
- Optional payload fields MUST have appropriate default values

**Vector Update Rules**:
- Vector embeddings MUST be immutable after initial creation
- Updates to vector representations MUST be handled as upsert operations
- Metadata updates MUST be allowed without changing the vector content
- Update operations MUST maintain cross-database consistency

**Delete/Replace Guarantees**:
- Vector deletion MUST also remove corresponding structured data records
- Bulk deletion operations MUST maintain atomicity where possible
- Replace operations MUST preserve ID consistency across databases
- Deletion operations MUST update all related cross-references

**Required Indexes**:
- Vector index MUST be created on all embedding vectors for similarity search
- Metadata indexes MUST be created on all searchable payload fields
- Composite indexes MUST be created for frequently combined search criteria
- Index maintenance MUST not impact query performance during operation

**Query Requirements for RAG Workflows**:
- Vector search MUST support configurable relevance thresholds
- Top-k retrieval MUST be supported with result scoring
- Query expansion capabilities MUST be available for enhanced recall
- Search result deduplication MUST be performed when necessary

**Integrity Constraints**:
- Vector dimensionality MUST remain consistent across all records
- Vector ID references MUST correspond to valid structured data records
- Mathematical properties of vectors MUST be preserved during operations
- Embedding parameters MUST remain consistent across all operations

**What Qdrant May NOT Store or Do**:
- Qdrant MUST NOT store raw text content except as minimal metadata payloads
- Qdrant MUST NOT perform relational queries or JOIN operations
- Qdrant MUST NOT store user account information or authentication data
- Qdrant MUST NOT maintain transactional relationships with other data

### 3.2 Neon / PostgreSQL

**User Accounts Schema Requirements**:
- Users table MUST include user_id as UUID primary key
- Users table MUST include creation_timestamp with timezone
- Users table MAY include profile_metadata as JSONB field
- Users table MUST enforce uniqueness constraints on appropriate fields

**Chat History Schema Requirements**:
- Chat history table MUST include chat_id as UUID primary key
- Chat history table MUST include user_id for user identification
- Chat history table MUST include query and response fields
- Chat history table MUST include source_chunks reference for citations
- Chat history table MUST include timestamp with timezone

**Document Schema Requirements**:
- Documents table MUST include document_id as UUID primary key
- Documents table MUST include document_reference field
- Documents table MUST include metadata fields for location tracking
- Documents table MUST include processing_version for pipeline tracking

**Settings/Config Tables**:
- Configuration tables MUST support key-value storage for system settings
- Settings MUST be versioned for change tracking
- Settings MUST include appropriate access controls
- Configuration validation MUST be performed during updates

**Logging/Audit Table Structure**:
- Audit logs MUST include operation_timestamp with timezone
- Audit logs MUST include user_id for access tracking
- Audit logs MUST include operation_type for categorization
- Audit logs MUST include resource_id for target identification
- Audit logs MUST preserve privacy-sensitive information appropriately

**Embedding Job Queue (Optional)**:
- Job queue tables MUST include job_id as UUID primary key
- Job queue tables MUST include status fields for processing tracking
- Job queue tables MUST include retry_count and failure_reason fields
- Job queue tables MUST include priority fields for processing order

**Foreign-Key Policies**:
- Foreign key constraints MUST be enforced with appropriate actions
- Referential integrity MUST be maintained across all related tables
- Cascade delete policies MUST be carefully considered for data integrity
- Constraint validation MUST be performed during all modification operations

**Indexing Requirements**:
- Primary key indexes MUST be created automatically
- Foreign key indexes MUST be created for join performance
- Search indexes MUST be created on frequently queried fields
- Composite indexes MUST be created for multi-field queries

**Data Normalization Level**:
- Database schema MUST follow third normal form (3NF) where appropriate
- Denormalization MAY be performed for query performance optimization
- Normalization rules MUST not compromise data integrity requirements
- Trade-offs between normalization and performance MUST be documented

**Concurrency Rules**:
- Database connections MUST use appropriate isolation levels
- Row-level locking MUST be available for high-contention scenarios
- Connection pooling MUST be implemented for resource optimization
- Deadlock detection and resolution MUST be handled by the database

**Transaction Requirements**:
- ACID properties MUST be maintained for all data operations
- Multi-statement transactions MUST be supported for complex operations
- Transaction boundaries MUST be clearly defined and documented
- Rollback capabilities MUST be available for all transactional operations

**Prohibited Behaviors**:
- PostgreSQL MUST NOT store vector embeddings (these belong in Qdrant)
- PostgreSQL MUST NOT perform ANN search operations (these belong in Qdrant)
- PostgreSQL MUST NOT store large text content that belongs in vector database
- PostgreSQL MUST NOT bypass the Database Subsystem's defined interfaces

## 4. Cross-Database Interaction Specifications

**Shared IDs**:
- UUID identifiers MUST match across related records in both databases
- ID generation MUST be consistent and coordinated between systems
- ID validation MUST be performed during cross-database operations
- ID consistency MUST be maintained during bulk operations

**Metadata Synchronization**:
- Chunk metadata in PostgreSQL MUST correspond to vectors in Qdrant
- Synchronization processes MUST handle failure scenarios gracefully
- Consistency checks MUST be performed regularly to detect drift
- Repair mechanisms MUST be available for detected inconsistencies

**Backend Request Contracts**:
- All requests to the Database Subsystem MUST go through defined interfaces
- Request formats MUST include appropriate metadata for routing
- Response formats MUST include consistent metadata across both databases
- Error responses MUST follow standardized error handling patterns

**Cross-Database Consistency Guarantees**:
- Related records MUST exist in both databases or neither
- Consistency validation MUST be performed during write operations
- Repair processes MUST be available for consistency failures
- Consistency checks MUST be performed during system startup

**Error-Handling Expectations**:
- Cross-database errors MUST be clearly identified in error messages
- Partial failure scenarios MUST be handled with appropriate rollbacks
- Error recovery processes MUST be defined and documented
- Circuit breaker patterns MUST be available for cascading failure prevention

**Retry Strategies**:
- Retry logic MUST be implemented for transient cross-database failures
- Exponential backoff MUST be used for retry operations
- Retry attempts MUST be limited to prevent system overload
- Retry contexts MUST preserve original request state

**No Duplication of Responsibilities**:
- Vector operations MUST be confined to Qdrant
- Structured data operations MUST be confined to PostgreSQL
- Query routing MUST follow established patterns
- Responsibility boundaries MUST be enforced through interface design

## 5. API Interface Requirements

**CRUD Operations**:
- Create operations MUST support single and batch record creation
- Read operations MUST support single and batch record retrieval
- Update operations MUST maintain cross-database consistency
- Delete operations MUST ensure complete record removal across databases

**Vector Search API**:
- Search operations MUST accept query vectors and return similarity results
- Search operations MUST support configurable result counts (top-k)
- Search operations MUST support metadata filtering during retrieval
- Search operations MUST return relevance scores with results

**Query Filtering Options**:
- Filter operations MUST support equality comparisons
- Filter operations MUST support range-based comparisons
- Filter operations MUST support text-based matching
- Filter operations MUST support boolean logic combinations

**Pagination Requirements**:
- Result sets MUST support offset-based pagination
- Result sets MUST support cursor-based pagination for better performance
- Page size MUST be configurable within reasonable limits
- Total count information MAY be provided for small result sets

**Batch Operations**:
- Batch insert operations MUST support bulk vector and metadata storage
- Batch query operations MUST support multiple record retrieval
- Batch update operations MUST maintain transactional consistency
- Batch operations MUST include progress tracking capabilities

**Error Formats**:
- All errors MUST include appropriate HTTP status codes
- Error responses MUST include descriptive error messages
- Error responses MUST include error codes for programmatic handling
- Error responses MUST not expose internal system details

**Timeout Requirements**:
- Read operations MUST timeout after 5 seconds if not completed
- Write operations MUST timeout after 10 seconds if not completed
- Search operations MUST timeout after 2 seconds if not completed
- Connection timeouts MUST be configurable and reasonable

**Required Responses**:
- Query responses MUST include appropriate metadata fields
- Search responses MUST include relevance scores and source information
- Error responses MUST include sufficient context for debugging
- Success responses MUST include confirmation of operation completion

**Required Metadata Fields**:
- All responses MUST include timestamp information
- All responses MUST include request correlation identifiers
- All responses MUST include resource identifiers where applicable
- All responses MUST include operation status information

## 6. Data Models

**User Model**:
- user_id (required): UUID identifier for the user account
- created_at (required): Timestamp with timezone for account creation
- email (optional): String field for email address with validation
- profile_metadata (optional): JSONB field for additional profile information
- preferences (optional): JSONB field for user preferences and settings
- is_active (required): Boolean field indicating account status
- last_login (optional): Timestamp with timezone for last login
- Constraints: user_id must be unique, email must follow standard format if present
- Relationships: May link to chat history records

**Chat Message Model**:
- message_id (required): UUID identifier for the message
- chat_id (required): UUID identifier for the conversation thread
- user_id (required): UUID identifier for the message author
- message_type (required): Enum field indicating query/response/system
- content (required): Text field containing the actual message content
- timestamp (required): Timestamp with timezone for message creation
- source_chunks (optional): JSONB field containing citation information
- parent_message_id (optional): UUID reference for message threading
- Constraints: message_id must be unique, proper ordering maintained
- Relationships: Links to user model and potentially other messages

**Document Model**:
- document_id (required): UUID identifier for the document
- document_reference (required): String field identifying the source document
- title (required): String field containing document title
- author (optional): String field for document author information
- created_at (required): Timestamp with timezone for document processing
- updated_at (required): Timestamp with timezone for last update
- metadata (optional): JSONB field for additional document metadata
- version (required): String field for document version tracking
- Constraints: document_reference must be unique per system
- Relationships: May link to chunk records

**Chunk Model**:
- chunk_id (required): UUID identifier matching vector record
- document_id (required): UUID reference to parent document
- content (required): Text field containing the chunk content
- page_reference (optional): Integer field for page location
- section_title (optional): String field for section context
- processing_version (required): String field for pipeline version
- embedding_status (required): Enum field indicating embedding generation status
- created_at (required): Timestamp with timezone for creation
- updated_at (required): Timestamp with timezone for last update
- Constraints: chunk_id must match corresponding vector record
- Relationships: Links to document model and vector record

**Vector Model**:
- vector_id (required): UUID identifier matching chunk record
- embedding_vector (required): Array of floating-point numbers (1024 dimensions)
- document_reference (required): String field for content identification
- chunk_content_preview (optional): Text field for content preview
- embedding_generation_params (optional): JSONB field for generation parameters
- indexed_at (required): Timestamp with timezone for indexing
- updated_at (required): Timestamp with timezone for last update
- Constraints: vector_id must match corresponding chunk record
- Relationships: Links to chunk model

**Settings Model**:
- setting_id (required): UUID identifier for the setting
- setting_key (required): String field for setting identifier
- setting_value (required): JSONB field for setting value (flexible type)
- setting_type (required): String field indicating value type
- created_at (required): Timestamp with timezone for creation
- updated_at (required): Timestamp with timezone for last update
- scope (optional): String field for setting scope (global/user-specific)
- Constraints: setting_key must be unique within scope
- Relationships: May link to user model for user-specific settings

**Audit Log Model**:
- log_id (required): UUID identifier for the log entry
- operation_type (required): String field indicating operation type
- resource_type (required): String field for resource category
- resource_id (required): String field for resource identifier
- user_id (optional): UUID reference to performing user
- operation_timestamp (required): Timestamp with timezone for operation
- details (optional): JSONB field for additional operation details
- ip_address (optional): String field for request origin
- user_agent (optional): String field for client information
- Constraints: Appropriate indexing for audit query performance
- Relationships: May link to user model if user identified

## 7. Security Requirements

**Encryption Expectations**:
- All data in transit MUST be encrypted using TLS 1.3 or higher
- Sensitive data at rest MUST be encrypted using AES-256 encryption
- Database connections MUST use encrypted communication channels
- API keys and credentials MUST be stored using secure encryption

**Access Control per Table/Collection**:
- User data tables MUST have row-level security controls
- Vector collection MUST have role-based access controls
- Audit logs MUST be write-only for application services
- Configuration tables MUST have limited write access

**Role-Based Permissions**:
- Database administrator role MUST have full access to all operations
- Application service role MUST have read/write access to operational data
- Read-only role MUST be available for analytics and reporting
- Anonymous access MUST be prohibited for all data operations

**Allowed Request Origins**:
- Database access MUST be restricted to backend services only
- Cross-origin requests MUST be validated against approved origins
- IP-based restrictions MAY be implemented for additional security
- Service-to-service authentication MUST be required

**Input Validation Assumptions**:
- All input MUST be validated at the application layer before database operations
- SQL injection prevention MUST be implemented through parameterized queries
- No raw user input MUST reach database operations directly
- Data type validation MUST be enforced at the database level

**Rate-Limiting Expectations**:
- Per-user query limits MUST be enforceable at the database level
- Bulk operation limits MUST be configurable and enforced
- Concurrent connection limits MUST be set and monitored
- Automated detection of unusual access patterns MUST be available

**Secure Query Patterns**:
- Parameterized queries MUST be used for all dynamic operations
- Stored procedures MAY be used for complex operations
- Query result size limits MUST be enforced to prevent resource exhaustion
- Prepared statements MUST be used where available for performance and security

**Compliance Requirements**:
- GDPR compliance MUST be maintained for user data handling
- Data retention policies MUST comply with applicable regulations
- Right to deletion MUST be supported for user data
- Data portability requirements MUST be accommodated

**Logging & Auditing Expectations**:
- All data access operations MUST be logged with appropriate detail
- Security-relevant operations MUST be logged with high priority
- Audit logs MUST be tamper-evident and securely stored
- Log retention policies MUST comply with regulatory requirements

**Secrets Handling Expectations**:
- Database credentials MUST be stored in secure secret management systems
- API keys MUST be rotated regularly according to security policies
- Connection string parameters MUST not be logged or exposed
- Certificate management MUST follow security best practices

## 8. Operational Requirements

**Deployment Expectations**:
- Database services MUST be deployed using cloud-managed services
- High availability configurations MUST be implemented where possible
- Geographic distribution MAY be implemented for performance requirements
- Deployment configurations MUST support environment-specific settings

**Performance Monitoring**:
- Query response times MUST be monitored and logged
- Database resource utilization MUST be continuously monitored
- Connection pool metrics MUST be tracked and alerted
- Slow query detection MUST be implemented and monitored

**Health Checks**:
- Database connectivity MUST be verified through health check endpoints
- Storage capacity MUST be monitored with appropriate alerts
- Query performance MUST be validated through synthetic health checks
- Cross-database consistency MUST be verified through regular checks

**Backup Frequency**:
- Full database backups MUST be performed daily for PostgreSQL
- Incremental backups MAY be performed more frequently
- Vector database backups MUST follow Qdrant's backup procedures
- Backup integrity MUST be verified through automated processes

**Alerting Rules**:
- High query response times MUST trigger performance alerts
- Low storage capacity MUST trigger capacity alerts
- Failed backup operations MUST trigger immediate alerts
- Security-relevant events MUST trigger security alerts

**Maintenance Window Policies**:
- Scheduled maintenance MUST be performed during low-usage periods
- Backup operations SHOULD be scheduled during maintenance windows
- Schema updates MUST be scheduled during planned maintenance
- System updates MUST follow appropriate change management procedures

**Migration Guidelines**:
- Schema migration processes MUST be tested in staging environments
- Data migration validation MUST be performed before and after migration
- Rollback procedures MUST be defined and tested for all migrations
- Migration scheduling MUST minimize impact on system availability

## 9. Anti-Corruption / Boundary Rules

**Backend cannot bypass the Database Subsystem**:
- All database access MUST go through the defined Database Subsystem interfaces
- Direct database connections from other subsystems MUST be prohibited
- Database credentials MUST not be exposed to other subsystems
- Raw SQL or query language MUST not be used outside the Database Subsystem

**Intelligence subsystem cannot modify the database**:
- The Intelligence Subsystem MUST only read from the database
- Write operations from the Intelligence Subsystem MUST be prohibited
- Direct database connections from the Intelligence Subsystem MUST be blocked
- The Intelligence Subsystem MUST go through the Backend for all database operations

**Embedding subsystem cannot read user data**:
- The Embedding Subsystem MUST only access document and chunk data
- User information MUST be prohibited from access by the Embedding Subsystem
- User data MUST be filtered out of any data processed by the Embedding Subsystem
- Cross-database access patterns MUST prevent unauthorized user data access

**Qdrant cannot be used for relational queries**:
- Qdrant MUST only store and retrieve vector embeddings with minimal metadata
- Complex relational queries MUST NOT be performed on Qdrant data
- JOIN operations between Qdrant and PostgreSQL MUST go through proper interfaces
- Qdrant MUST NOT store data that belongs in the structured database

**Postgres cannot be used as vector index**:
- PostgreSQL MUST NOT store vector embeddings for similarity search
- Vector operations MUST NOT be performed in PostgreSQL
- ANN search capabilities MUST NOT be implemented in PostgreSQL
- Vector-specific indexes MUST NOT be created in PostgreSQL

**No direct inter-database writes**:
- Cross-database writes MUST be coordinated through the Database Subsystem
- Direct communication between Qdrant and PostgreSQL MUST be prohibited
- Atomicity of cross-database operations MUST be maintained through proper interfaces
- Consistency checks MUST validate proper inter-database relationships

**No circular dependencies**:
- The Database Subsystem MUST not depend on other subsystems for operation
- Database access patterns MUST not create circular call chains
- Interface contracts MUST be designed to prevent dependency cycles
- System architecture MUST maintain clear directional data flow

## 10. Future Extensions

**Multi-vector embeddings**:
- Support for multiple embedding models MAY be implemented
- Vector comparison across different models MAY be supported
- Model-specific indexes MAY be created for performance
- Migration paths for embedding model updates MUST be planned

**Semantic cache**:
- High-frequency query caching MAY be implemented
- Vector similarity caching MAY improve response times
- Cache invalidation strategies MUST maintain data consistency
- Distributed caching MAY be implemented for scalability

**Vector re-ranking database**:
- Secondary database for re-ranking results MAY be added
- Machine learning model integration MAY enhance re-ranking
- Additional metadata for re-ranking MAY be stored
- Performance optimization MAY require separate re-ranking infrastructure

**Additional metadata fields**:
- Enhanced document metadata MAY be supported
- Content categorization fields MAY be added
- User interaction metadata MAY be tracked
- Performance metrics metadata MAY be stored

**Multi-tenant support**:
- Isolated data storage per tenant MAY be implemented
- Tenant-specific access controls MAY be required
- Resource allocation per tenant MAY be needed
- Tenant migration tools MAY be developed

**Hybrid search**:
- Keyword-based search MAY be combined with vector search
- Full-text search capabilities MAY be integrated
- Multi-modal search MAY be supported
- Result ranking algorithms MAY be enhanced

**Sharding or distributed architecture**:
- Horizontal partitioning MAY be implemented for scalability
- Distributed query processing MAY be required
- Cross-shard consistency mechanisms MAY be needed
- Load balancing across database nodes MAY be implemented

## Outcome

- ✅ Impact: Created comprehensive specification for Database Subsystem
- 🧪 Tests: null
- 📁 Files: Created Chatbot_specs/database_specs/specification.md
- 🔁 Next prompts: Ready for planning and implementation phases
- 🧠 Reflection: Specification fully addresses all required sections and aligns with constitution

## Evaluation notes (flywheel)

- Failure modes observed: null
- Graders run and results (PASS/FAIL): null
- Prompt variant (if applicable): null
- Next experiment (smallest change to try): null