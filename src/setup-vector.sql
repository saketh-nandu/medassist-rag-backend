-- ============================================================
-- Real RAG: pgvector setup
-- Run this in Supabase SQL Editor ONCE
-- ============================================================

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create chunks table (stores chunked + embedded documents)
DROP TABLE IF EXISTS medical_chunks CASCADE;

CREATE TABLE medical_chunks (
  id          BIGSERIAL PRIMARY KEY,
  doc_id      TEXT NOT NULL,          -- source document identifier
  source      TEXT NOT NULL,          -- DSP / MedQuAD / BuiltinKnowledge
  category    TEXT NOT NULL,          -- disease / qa / precaution
  title       TEXT NOT NULL,          -- document title
  chunk_index INT  NOT NULL DEFAULT 0,-- chunk position within document
  content     TEXT NOT NULL,          -- the actual text chunk
  embedding   vector(384),            -- all-MiniLM-L6-v2 produces 384-dim
  metadata    JSONB DEFAULT '{}',
  created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- 3. IVFFlat index for fast approximate nearest-neighbour search
--    (cosine distance — best for semantic similarity)
CREATE INDEX medical_chunks_embedding_idx
  ON medical_chunks
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

CREATE INDEX medical_chunks_source_idx ON medical_chunks(source);
CREATE INDEX medical_chunks_category_idx ON medical_chunks(category);

-- 4. Vector similarity search function
--    Returns top-k chunks by cosine similarity to query embedding
CREATE OR REPLACE FUNCTION match_chunks(
  query_embedding vector(384),
  match_count     INT     DEFAULT 6,
  filter_category TEXT    DEFAULT NULL
)
RETURNS TABLE (
  id          BIGINT,
  doc_id      TEXT,
  source      TEXT,
  category    TEXT,
  title       TEXT,
  chunk_index INT,
  content     TEXT,
  metadata    JSONB,
  similarity  FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    mc.id,
    mc.doc_id,
    mc.source,
    mc.category,
    mc.title,
    mc.chunk_index,
    mc.content,
    mc.metadata,
    1 - (mc.embedding <=> query_embedding) AS similarity
  FROM medical_chunks mc
  WHERE
    mc.embedding IS NOT NULL
    AND (filter_category IS NULL OR mc.category = filter_category)
  ORDER BY mc.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;
