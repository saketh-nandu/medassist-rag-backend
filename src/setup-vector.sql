-- ============================================================
-- Enhanced RAG: pgvector setup with HNSW + FTS
-- Run this in Supabase SQL Editor ONCE
-- ============================================================

-- 1. Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create chunks table
DROP TABLE IF EXISTS medical_chunks CASCADE;

CREATE TABLE medical_chunks (
  id          BIGSERIAL PRIMARY KEY,
  doc_id      TEXT NOT NULL,
  source      TEXT NOT NULL,
  category    TEXT NOT NULL,
  title       TEXT NOT NULL,
  chunk_index INT  NOT NULL DEFAULT 0,
  content     TEXT NOT NULL,
  embedding   vector(384),            -- all-MiniLM-L6-v2 → 384 dims
  metadata    JSONB DEFAULT '{}',
  created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- 3. HNSW index — faster and more accurate than IVFFlat
--    m=16 (connections per node), ef_construction=64 (build quality)
--    cosine distance for semantic similarity
CREATE INDEX medical_chunks_hnsw_idx
  ON medical_chunks
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- 4. Full-text search index for hybrid retrieval
ALTER TABLE medical_chunks
  ADD COLUMN IF NOT EXISTS content_tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('english', coalesce(title, '') || ' ' || coalesce(content, ''))) STORED;

CREATE INDEX medical_chunks_fts_idx
  ON medical_chunks USING gin(content_tsv);

-- 5. Supporting indexes
CREATE INDEX medical_chunks_source_idx   ON medical_chunks(source);
CREATE INDEX medical_chunks_category_idx ON medical_chunks(category);
CREATE INDEX medical_chunks_doc_id_idx   ON medical_chunks(doc_id);

-- 6. Vector similarity search function (primary retrieval path)
CREATE OR REPLACE FUNCTION match_chunks(
  query_embedding vector(384),
  match_count     INT     DEFAULT 8,
  filter_category TEXT    DEFAULT NULL,
  similarity_threshold FLOAT DEFAULT 0.3
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
    AND 1 - (mc.embedding <=> query_embedding) >= similarity_threshold
  ORDER BY mc.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- 7. Full-text search function (keyword retrieval path)
CREATE OR REPLACE FUNCTION search_chunks_fts(
  query_text  TEXT,
  match_count INT DEFAULT 8
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
  rank        FLOAT
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
    ts_rank(mc.content_tsv, websearch_to_tsquery('english', query_text))::FLOAT AS rank
  FROM medical_chunks mc
  WHERE mc.content_tsv @@ websearch_to_tsquery('english', query_text)
  ORDER BY rank DESC
  LIMIT match_count;
END;
$$;

-- 8. Hybrid search function (vector + FTS combined via RRF in SQL)
--    Useful if you want to do RRF server-side instead of in JS
CREATE OR REPLACE FUNCTION hybrid_search(
  query_embedding vector(384),
  query_text      TEXT,
  match_count     INT   DEFAULT 8,
  rrf_k           INT   DEFAULT 60
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
  rrf_score   FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  WITH vector_results AS (
    SELECT mc.id, ROW_NUMBER() OVER (ORDER BY mc.embedding <=> query_embedding) AS rank
    FROM medical_chunks mc
    WHERE mc.embedding IS NOT NULL
    LIMIT 50
  ),
  fts_results AS (
    SELECT mc.id, ROW_NUMBER() OVER (ORDER BY ts_rank(mc.content_tsv, websearch_to_tsquery('english', query_text)) DESC) AS rank
    FROM medical_chunks mc
    WHERE mc.content_tsv @@ websearch_to_tsquery('english', query_text)
    LIMIT 50
  ),
  rrf AS (
    SELECT
      COALESCE(v.id, f.id) AS id,
      COALESCE(1.0 / (rrf_k + v.rank), 0) + COALESCE(1.0 / (rrf_k + f.rank), 0) AS score
    FROM vector_results v
    FULL OUTER JOIN fts_results f ON v.id = f.id
  )
  SELECT
    mc.id, mc.doc_id, mc.source, mc.category, mc.title,
    mc.chunk_index, mc.content, mc.metadata,
    rrf.score AS rrf_score
  FROM rrf
  JOIN medical_chunks mc ON mc.id = rrf.id
  ORDER BY rrf.score DESC
  LIMIT match_count;
END;
$$;
