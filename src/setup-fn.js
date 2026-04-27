/**
 * setup-fn.js
 * Creates the match_chunks pgvector function in Supabase.
 * Run once: node src/setup-fn.js
 */
import dotenv from 'dotenv';
dotenv.config();

const SUPABASE_URL = process.env.SUPABASE_URL;
const SERVICE_KEY  = process.env.SUPABASE_SERVICE_KEY;

// Supabase exposes a /rest/v1/rpc endpoint but not raw SQL.
// We use the pg endpoint available on the project's REST API.
// The correct way is via the Supabase Management API or direct pg connection.
// Since we only have the anon/service key, we use the SQL-over-HTTP trick:
// POST to /rest/v1/ with a raw query via the pg extension if enabled,
// OR we use the Supabase "sql" endpoint (available in newer Supabase versions).

const sql = `
CREATE EXTENSION IF NOT EXISTS vector;

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

// Try Supabase SQL endpoint (available via service role)
const projectRef = SUPABASE_URL.match(/https:\/\/([^.]+)\.supabase\.co/)?.[1];
if (!projectRef) {
  console.error('Could not extract project ref from SUPABASE_URL');
  process.exit(1);
}

const endpoint = `https://api.supabase.com/v1/projects/${projectRef}/database/query`;

console.log(`\nCreating match_chunks function via Supabase Management API...`);
console.log(`Project: ${projectRef}`);

const res = await fetch(endpoint, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${SERVICE_KEY}`,
  },
  body: JSON.stringify({ query: sql }),
});

const text = await res.text();
console.log(`Status: ${res.status}`);

if (res.ok) {
  console.log('✅ match_chunks function created successfully!');
} else {
  console.log('Response:', text.slice(0, 500));
  console.log('\n⚠️  Management API failed. Please run this SQL manually in Supabase SQL Editor:');
  console.log('https://supabase.com/dashboard/project/' + projectRef + '/sql/new');
  console.log('\n--- COPY THIS SQL ---');
  console.log(sql);
  console.log('--- END SQL ---');
}
