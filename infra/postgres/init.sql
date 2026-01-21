-- Research Agent Database Schema
-- Initial schema for research logs and metadata

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Research queries table
CREATE TABLE IF NOT EXISTS research_queries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query TEXT NOT NULL,
    domains TEXT[] NOT NULL DEFAULT '{}',
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    cost_usd DECIMAL(10, 6) DEFAULT 0,
    tokens_used INTEGER DEFAULT 0
);

-- Research logs table (for debugging and analysis)
CREATE TABLE IF NOT EXISTS research_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_id UUID REFERENCES research_queries(id) ON DELETE CASCADE,
    step VARCHAR(100) NOT NULL,
    node_name VARCHAR(100),
    input_data JSONB,
    output_data JSONB,
    duration_ms INTEGER,
    tokens_used INTEGER DEFAULT 0,
    cost_usd DECIMAL(10, 6) DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Generated papers table
CREATE TABLE IF NOT EXISTS papers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_id UUID REFERENCES research_queries(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    abstract TEXT,
    latex_content TEXT NOT NULL,
    bibtex_content TEXT,
    pdf_path TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Sources table (papers, articles referenced)
CREATE TABLE IF NOT EXISTS sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_id UUID REFERENCES research_queries(id) ON DELETE CASCADE,
    source_type VARCHAR(50) NOT NULL, -- 'arxiv', 'web', 'paper'
    title TEXT,
    url TEXT,
    arxiv_id TEXT,
    authors TEXT[],
    abstract TEXT,
    bibtex_key TEXT,
    bibtex_entry TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_research_queries_status ON research_queries(status);
CREATE INDEX IF NOT EXISTS idx_research_queries_created_at ON research_queries(created_at);
CREATE INDEX IF NOT EXISTS idx_research_logs_query_id ON research_logs(query_id);
CREATE INDEX IF NOT EXISTS idx_research_logs_step ON research_logs(step);
CREATE INDEX IF NOT EXISTS idx_papers_query_id ON papers(query_id);
CREATE INDEX IF NOT EXISTS idx_sources_query_id ON sources(query_id);
CREATE INDEX IF NOT EXISTS idx_sources_arxiv_id ON sources(arxiv_id);

-- Updated at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to research_queries
DROP TRIGGER IF EXISTS update_research_queries_updated_at ON research_queries;
CREATE TRIGGER update_research_queries_updated_at
    BEFORE UPDATE ON research_queries
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
