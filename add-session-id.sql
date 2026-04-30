-- ============================================================
-- Add session_id column to chat_messages and health_reports
-- Run this in Supabase SQL Editor
-- ============================================================

-- 1. Add session_id to chat_messages
ALTER TABLE chat_messages
  ADD COLUMN IF NOT EXISTS session_id TEXT;

-- Create index for fast session lookups
CREATE INDEX IF NOT EXISTS chat_messages_session_idx
  ON chat_messages(session_id);

-- 2. Add session_id to health_reports
ALTER TABLE health_reports
  ADD COLUMN IF NOT EXISTS session_id TEXT;

-- Create index
CREATE INDEX IF NOT EXISTS health_reports_session_idx
  ON health_reports(session_id);

-- 3. Backfill existing messages with a default session per user
-- (groups all existing messages into one "legacy" session per user)
UPDATE chat_messages
SET session_id = 'legacy_session_' || user_id
WHERE session_id IS NULL;

UPDATE health_reports
SET session_id = 'legacy_session_' || user_id
WHERE session_id IS NULL;
