/**
 * create-fn.js
 * Creates the match_chunks SQL function in Supabase via pg REST.
 * Run once: node src/create-fn.js
 */
import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';
dotenv.config();

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY
);

const sql = `
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
`;

// Supabase doesn't expose raw SQL via JS client directly —
// use the pg REST endpoint with service key
const res = await fetch(`${process.env.SUPABASE_URL}/rest/v1/rpc/`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'apikey': process.env.SUPABASE_SERVICE_KEY,
    'Authorization': `Bearer ${process.env.SUPABASE_SERVICE_KEY}`,
  },
});

// Use the management API instead
const mgmtRes = await fetch(
  `${process.env.SUPABASE_URL.replace('.supabase.co', '')}.supabase.co/pg/query`,
  {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'apikey': process.env.SUPABASE_SERVICE_KEY,
      'Authorization': `Bearer ${process.env.SUPABASE_SERVICE_KEY}`,
    },
    body: JSON.stringify({ query: sql }),
  }
);

console.log('Status:', mgmtRes.status);
const body = await mgmtRes.text();
console.log('Response:', body.slice(0, 300));
