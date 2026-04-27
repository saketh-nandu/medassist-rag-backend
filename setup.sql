-- ============================================================
-- HosFind RAG - Run this in Supabase SQL Editor
-- https://supabase.com/dashboard/project/smybetwoumsiqtwpirvi/sql/new
-- ============================================================

-- Enable trigram extension
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Drop and recreate
DROP TABLE IF EXISTS medical_knowledge CASCADE;

CREATE TABLE medical_knowledge (
  id          BIGSERIAL PRIMARY KEY,
  source      TEXT NOT NULL,
  category    TEXT NOT NULL,
  title       TEXT NOT NULL,
  content     TEXT NOT NULL,
  symptoms    TEXT[] DEFAULT '{}',
  precautions TEXT[] DEFAULT '{}',
  metadata    JSONB DEFAULT '{}',
  fts         TSVECTOR,
  created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- GIN index for fast full-text search
CREATE INDEX medical_knowledge_fts_idx
  ON medical_knowledge USING GIN(fts);

CREATE INDEX medical_knowledge_category_idx
  ON medical_knowledge(category);

CREATE INDEX medical_knowledge_title_idx
  ON medical_knowledge(lower(title));

-- Trigger function to auto-update fts column on insert/update
CREATE OR REPLACE FUNCTION medical_knowledge_fts_update()
RETURNS TRIGGER AS $$
BEGIN
  NEW.fts := to_tsvector('english',
    coalesce(NEW.title, '') || ' ' ||
    coalesce(NEW.content, '') || ' ' ||
    coalesce(array_to_string(NEW.symptoms, ' '), '')
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER medical_knowledge_fts_trigger
  BEFORE INSERT OR UPDATE ON medical_knowledge
  FOR EACH ROW EXECUTE FUNCTION medical_knowledge_fts_update();

-- Full-text search function
CREATE OR REPLACE FUNCTION search_medical(
  query_text   TEXT,
  match_count  INT DEFAULT 8,
  filter_cat   TEXT DEFAULT NULL
)
RETURNS TABLE (
  id          BIGINT,
  source      TEXT,
  category    TEXT,
  title       TEXT,
  content     TEXT,
  symptoms    TEXT[],
  precautions TEXT[],
  metadata    JSONB,
  rank        FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    mk.id,
    mk.source,
    mk.category,
    mk.title,
    mk.content,
    mk.symptoms,
    mk.precautions,
    mk.metadata,
    ts_rank(mk.fts, websearch_to_tsquery('english', query_text))::FLOAT AS rank
  FROM medical_knowledge mk
  WHERE
    mk.fts @@ websearch_to_tsquery('english', query_text)
    AND (filter_cat IS NULL OR mk.category = filter_cat)
  ORDER BY rank DESC
  LIMIT match_count;
END;
$$;

SELECT 'Setup complete!' AS status;

-- ============================================================
-- Reminders table — persists medicine reminders per user
-- ============================================================

CREATE TABLE IF NOT EXISTS reminders (
  id            TEXT PRIMARY KEY,          -- client-generated UUID
  user_id       UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  medicine_name TEXT NOT NULL,
  dosage        TEXT NOT NULL DEFAULT '1 tablet',
  frequency     TEXT NOT NULL DEFAULT 'Daily',
  time_slots    TEXT[] NOT NULL DEFAULT '{"08:00 AM"}',
  status        TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','paused')),
  taken_today   BOOLEAN NOT NULL DEFAULT FALSE,
  taken_date    DATE,                       -- date when taken_today was last set
  created_at    TIMESTAMPTZ DEFAULT NOW(),
  updated_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Each user can only see their own reminders
ALTER TABLE reminders ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can manage own reminders"
  ON reminders FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- Auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN NEW.updated_at = NOW(); RETURN NEW; END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER reminders_updated_at
  BEFORE UPDATE ON reminders
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- Reset taken_today daily (call this via a cron or on app load)
CREATE OR REPLACE FUNCTION reset_taken_today()
RETURNS void AS $$
BEGIN
  UPDATE reminders
  SET taken_today = FALSE
  WHERE taken_date < CURRENT_DATE OR taken_date IS NULL;
END;
$$ LANGUAGE plpgsql;

SELECT 'Reminders table created!' AS status;

-- ============================================================
-- Chat messages table — persists full conversation history
-- ============================================================

CREATE TABLE IF NOT EXISTS chat_messages (
  id              TEXT PRIMARY KEY,
  user_id         UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
  content         TEXT NOT NULL,
  severity        TEXT CHECK (severity IN ('low', 'medium', 'high')),
  conditions      TEXT[] DEFAULT '{}',
  suggested_actions TEXT[] DEFAULT '{}',
  affected_areas  TEXT[] DEFAULT '{}',
  is_image        BOOLEAN DEFAULT FALSE,
  created_at      TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users manage own messages" ON chat_messages
  FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

CREATE INDEX chat_messages_user_idx ON chat_messages(user_id, created_at DESC);

-- ============================================================
-- Health reports table — AI-generated symptom reports
-- ============================================================

CREATE TABLE IF NOT EXISTS health_reports (
  id              BIGSERIAL PRIMARY KEY,
  user_id         UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  title           TEXT NOT NULL,
  conditions      TEXT[] DEFAULT '{}',
  severity        TEXT CHECK (severity IN ('low', 'medium', 'high')),
  suggested_actions TEXT[] DEFAULT '{}',
  affected_areas  TEXT[] DEFAULT '{}',
  chat_message_id TEXT REFERENCES chat_messages(id) ON DELETE SET NULL,
  created_at      TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE health_reports ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users manage own reports" ON health_reports
  FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

CREATE INDEX health_reports_user_idx ON health_reports(user_id, created_at DESC);

SELECT 'Chat and reports tables created!' AS status;

-- Add adherence tracking columns to reminders (run if table already exists)
ALTER TABLE reminders ADD COLUMN IF NOT EXISTS total_doses   INT NOT NULL DEFAULT 14;
ALTER TABLE reminders ADD COLUMN IF NOT EXISTS taken_doses   INT NOT NULL DEFAULT 0;
ALTER TABLE reminders ADD COLUMN IF NOT EXISTS start_date    DATE DEFAULT CURRENT_DATE;
ALTER TABLE reminders ADD COLUMN IF NOT EXISTS duration_days INT NOT NULL DEFAULT 7;
ALTER TABLE reminders ADD COLUMN IF NOT EXISTS notification_ids TEXT[] DEFAULT '{}';

SELECT 'Reminders columns updated!' AS status;
