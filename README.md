# HosFind RAG Backend

Medical AI backend using **Supabase full-text search** — no LLM, no embeddings, no paid APIs.

## How it works

```
User Query
    ↓
Supabase PostgreSQL full-text search (tsvector)
    ↓
Top 6 matching medical documents
    ↓
Template-based response builder
    ↓
Structured JSON → App
```

**Zero external APIs. Zero cost. 100% local + Supabase.**

## Setup

### 1. Install dependencies
```bash
cd rag-backend
npm install
```

### 2. Configure environment
```bash
cp .env.example .env
```

Fill in `.env`:
- `SUPABASE_URL` — your project URL (already set)
- `SUPABASE_SERVICE_KEY` — from Supabase Dashboard → Settings → API → `service_role` key

### 3. Create database schema

**Option A: Supabase SQL Editor (recommended)**
1. Go to https://supabase.com/dashboard/project/smybetwoumsiqtwpirvi/sql/new
2. Copy the entire contents of `setup.sql`
3. Paste and click "Run"

**Option B: Verify script**
```bash
node src/setupDb.js
```
This just tests the connection. You still need to run `setup.sql` manually.

### 4. Enable pg_trgm extension
- Dashboard → Database → Extensions → search `pg_trgm` → Enable

### 5. Ingest datasets
```bash
# Dry run first (preview counts)
npm run dry-run

# Full ingestion (8,273 rows)
npm run ingest
```

This loads:
- **MedQuAD**: 8,192 medical Q&A pairs from NIH
- **DSP**: 42 diseases with symptoms + descriptions + precautions
- **Symptom2Disease**: 24 diseases with patient symptom descriptions
- **Built-in**: 15 common diseases/injuries

### 6. Start the server
```bash
npm run dev    # development with auto-reload
npm start      # production
```

Server runs on `http://localhost:3001`

## API Endpoints

### `POST /api/chat`
Main chat endpoint.

**Request:**
```json
{ "message": "I have severe headache and fever for 2 days" }
```

**Response:**
```json
{
  "content": "Based on your symptoms, this may be related to **Migraine** or **Dengue Fever**...",
  "conditions": ["Migraine", "Dengue Fever", "Typhoid"],
  "suggestedActions": ["Stay hydrated", "Take paracetamol", "Consult a doctor"],
  "severity": "medium",
  "affectedAreas": ["head", "chest"],
  "isEmergency": false
}
```

### `GET /api/health`
Server status + knowledge base size.

### `GET /api/stats`
Knowledge base breakdown by source and category.

### `POST /api/search`
Direct full-text search for testing.

## Cost

**$0.00** — Everything runs on Supabase free tier (500MB database).

## Datasets Used

All datasets are in `../project/`:
- `MedQuAD-master/` — 11,263 XML files from NIH
- `DSP/` — Disease-Symptom-Precaution dataset (Kaggle)
- `Symptom2Disease.csv` — 1,200 symptom descriptions (Kaggle)

No downloads needed — already in your project folder.
