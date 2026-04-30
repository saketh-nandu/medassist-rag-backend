-- Vector similarity search function for MedAssist RAG
CREATE OR REPLACE FUNCTION match_chunks(
  query_embedding vector(384),
  match_count int DEFAULT 10,
  similarity_threshold float DEFAULT 0.1
)
RETURNS TABLE (
  id bigint,
  doc_id text,
  source text,
  category text,
  title text,
  chunk_index integer,
  content text,
  metadata jsonb,
  embedding vector(384),
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    medical_chunks.id,
    medical_chunks.doc_id,
    medical_chunks.source,
    medical_chunks.category,
    medical_chunks.title,
    medical_chunks.chunk_index,
    medical_chunks.content,
    medical_chunks.metadata,
    medical_chunks.embedding,
    1 - (medical_chunks.embedding <=> query_embedding) AS similarity
  FROM medical_chunks
  WHERE 1 - (medical_chunks.embedding <=> query_embedding) > similarity_threshold
  ORDER BY medical_chunks.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;