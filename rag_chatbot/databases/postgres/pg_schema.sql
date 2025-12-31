-- PostgreSQL Schema for RAG Chatbot System

-- Chunks table: Stores chunk metadata including chunk_id (UUID), document_reference, page_reference, section_title, chunk_text, embedding_id, creation timestamps, and processing_version
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id UUID PRIMARY KEY,
    document_reference VARCHAR(255) NOT NULL,
    page_reference INTEGER,
    section_title VARCHAR(255),
    chunk_text TEXT NOT NULL,
    embedding_id UUID NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processing_version VARCHAR(50) NOT NULL,
    metadata JSONB DEFAULT '{}'
);

-- Indexes for chunks table
CREATE INDEX IF NOT EXISTS idx_chunks_document_ref ON chunks(document_reference);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_id ON chunks(embedding_id);
CREATE INDEX IF NOT EXISTS idx_chunks_created_at ON chunks(created_at);

-- Logs table: Record system events, user queries, retrieved chunks, responses, and retrieval modes with timestamps
CREATE TABLE IF NOT EXISTS logs (
    log_id UUID PRIMARY KEY,
    user_query TEXT NOT NULL,
    retrieved_chunks JSONB NOT NULL,
    response TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    retrieval_mode VARCHAR(50) NOT NULL  -- 'full_book' or 'selected_text'
);

-- Indexes for logs table
CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_logs_retrieval_mode ON logs(retrieval_mode);

-- Chat history table: Maintain conversation history with user_id, queries, responses, source_chunks, and timestamps
CREATE TABLE IF NOT EXISTS chat_history (
    chat_id UUID PRIMARY KEY,
    user_id UUID,
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    source_chunks JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for chat_history table
CREATE INDEX IF NOT EXISTS idx_chat_history_user_id ON chat_history(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_history_timestamp ON chat_history(timestamp);

-- Users table: Store user accounts with unique identifiers, creation timestamps, and optional metadata
CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    email VARCHAR(255) UNIQUE,
    profile_metadata JSONB DEFAULT '{}',
    preferences JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    last_login TIMESTAMP WITH TIME ZONE
);

-- Indexes for users table
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);

-- Audit logs table: Track system operations and security events
CREATE TABLE IF NOT EXISTS audit_logs (
    log_id UUID PRIMARY KEY,
    operation_type VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100) NOT NULL,
    resource_id VARCHAR(255) NOT NULL,
    user_id UUID,
    operation_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    details JSONB,
    ip_address INET,
    user_agent TEXT
);

-- Indexes for audit_logs table
CREATE INDEX IF NOT EXISTS idx_audit_logs_operation_type ON audit_logs(operation_type);
CREATE INDEX IF NOT EXISTS idx_audit_logs_resource_id ON audit_logs(resource_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_operation_timestamp ON audit_logs(operation_timestamp);

-- Settings table: Store system settings and user preferences
CREATE TABLE IF NOT EXISTS settings (
    setting_id UUID PRIMARY KEY,
    setting_key VARCHAR(255) NOT NULL UNIQUE,
    setting_value JSONB NOT NULL,
    setting_type VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    scope VARCHAR(50) DEFAULT 'global'  -- 'global' or 'user'
);

-- Indexes for settings table
CREATE INDEX IF NOT EXISTS idx_settings_key ON settings(setting_key);
CREATE INDEX IF NOT EXISTS idx_settings_scope ON settings(scope);

-- Embedding jobs queue table: Optional table for tracking embedding generation jobs
CREATE TABLE IF NOT EXISTS embedding_jobs (
    job_id UUID PRIMARY KEY,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',  -- 'pending', 'processing', 'completed', 'failed'
    document_reference VARCHAR(255) NOT NULL,
    chunk_count INTEGER NOT NULL,
    processed_count INTEGER DEFAULT 0,
    retry_count INTEGER DEFAULT 0,
    failure_reason TEXT,
    priority INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for embedding_jobs table
CREATE INDEX IF NOT EXISTS idx_embedding_jobs_status ON embedding_jobs(status);
CREATE INDEX IF NOT EXISTS idx_embedding_jobs_document_ref ON embedding_jobs(document_reference);
CREATE INDEX IF NOT EXISTS idx_embedding_jobs_priority ON embedding_jobs(priority DESC, created_at);

-- Foreign key constraints
ALTER TABLE chat_history ADD CONSTRAINT fk_chat_history_user_id
    FOREIGN KEY (user_id) REFERENCES users(user_id)
    ON DELETE SET NULL;

ALTER TABLE audit_logs ADD CONSTRAINT fk_audit_logs_user_id
    FOREIGN KEY (user_id) REFERENCES users(user_id)
    ON DELETE SET NULL;

-- Update trigger function for updating 'updated_at' timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS '
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
' language 'plpgsql';

-- Attach the trigger to tables that have updated_at columns
DROP TRIGGER IF EXISTS update_chunks_updated_at ON chunks;
CREATE TRIGGER update_chunks_updated_at
    BEFORE UPDATE ON chunks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_settings_updated_at ON settings;
CREATE TRIGGER update_settings_updated_at
    BEFORE UPDATE ON settings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- RLS (Row Level Security) - disabled by default, can be enabled for multi-tenant scenarios
-- ALTER TABLE chat_history ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE logs ENABLE ROW LEVEL SECURITY;