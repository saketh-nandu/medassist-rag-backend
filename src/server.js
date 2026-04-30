/**
 * server.js — MedAssist RAG Backend
 * Deployed on Render (or any Node host)
 *
 * Stack:
 *   - Express
 *   - Supabase pgvector (11,458 embedded chunks)
 *   - all-MiniLM-L6-v2 for query embedding (via @xenova/transformers)
 *   - Groq llama-3.3-70b-versatile for generation
 */

import express from 'express';
import cors from 'cors';
import { createClient } from '@supabase/supabase-js';
import { ragQuery } from './rag.js';
import { warmupEmbedder } from './embed.js';
import { evaluateModel, getModelMetrics } from './model-evaluation.js';
import dotenv from 'dotenv';
dotenv.config();

const app  = express();
const PORT = process.env.PORT || 3001;

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY
);

app.use(cors({ origin: '*' }));
app.use(express.json({ limit: '1mb' }));
app.use((req, _res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
  next();
});

// ─── HEALTH ───────────────────────────────────────────────────────────────────

app.get('/', (_req, res) => res.json({ status: 'ok', service: 'MedAssist RAG Backend' }));

app.get('/api/health', async (_req, res) => {
  const { count } = await supabase
    .from('medical_chunks')
    .select('*', { count: 'exact', head: true });

  res.json({
    status: 'ok',
    mode: 'pgvector + all-MiniLM-L6-v2 + Groq llama-3.3-70b-versatile (hybrid RAG)',
    chunks: count || 0,
  });
});

// ─── CHAT ─────────────────────────────────────────────────────────────────────

app.post('/api/chat', async (req, res) => {
  const startTime = Date.now();
  const { message, history = [] } = req.body;

  console.log('\n🚀 ===== NEW RAG REQUEST =====');
  console.log(`📝 User Query: "${message}"`);
  console.log(`📚 History Length: ${history?.length || 0} messages`);
  console.log(`⏰ Timestamp: ${new Date().toISOString()}`);

  if (!message || typeof message !== 'string' || !message.trim()) {
    return res.status(400).json({ error: 'message is required' });
  }
  if (message.length > 2000) {
    return res.status(400).json({ error: 'message too long (max 2000 chars)' });
  }

  try {
    const response = await ragQuery(message.trim(), history);
    
    const processingTime = Date.now() - startTime;
    
    console.log(`⚡ Total Processing Time: ${processingTime}ms`);
    console.log(`📤 Response Size: ${JSON.stringify(response).length} bytes`);
    console.log('=======================================\n');
    
    res.json({
      ...response,
      metadata: {
        processingTime,
        timestamp: new Date().toISOString(),
        version: '1.0.0'
      }
    });
  } catch (err) {
    const processingTime = Date.now() - startTime;
    
    console.error('❌ ===== RAG REQUEST FAILED =====');
    console.error(`💥 Error: ${err.message}`);
    console.error(`⏱️  Failed after: ${processingTime}ms`);
    console.error(`📍 Stack: ${err.stack}`);
    console.error('=======================================\n');
    
    res.status(500).json({
      content: 'An error occurred. Please try again or consult a doctor directly.',
      conditions: [],
      suggestedActions: ['Consult a doctor directly', 'Call 108 for emergencies'],
      severity: 'low',
      affectedAreas: [],
      isEmergency: false,
      metadata: {
        processingTime,
        timestamp: new Date().toISOString(),
        error: true
      }
    });
  }
});

// ─── MODEL EVALUATION ────────────────────────────────────────────────────────

app.get('/api/model/metrics', async (_req, res) => {
  try {
    const metrics = await getModelMetrics();
    res.json({
      status: 'success',
      metrics,
      timestamp: new Date().toISOString()
    });
  } catch (err) {
    console.error('[Model Metrics] Error:', err.message);
    res.status(500).json({ error: 'Failed to get model metrics' });
  }
});

app.post('/api/model/evaluate', async (_req, res) => {
  console.log('\n🧠 ===== MODEL EVALUATION REQUEST =====');
  console.log(`⏰ Started at: ${new Date().toISOString()}`);
  
  try {
    const evaluation = await evaluateModel();
    
    console.log('📊 ===== EVALUATION COMPLETE =====');
    console.log(`🎯 Overall Score: ${(evaluation.overall_score * 100).toFixed(1)}%`);
    console.log(`🔍 Retrieval: ${(evaluation.retrieval_metrics.average_score * 100).toFixed(1)}%`);
    console.log(`🤖 Generation: ${(evaluation.generation_metrics.average_score * 100).toFixed(1)}%`);
    console.log(`⚡ Efficiency: ${(evaluation.efficiency_metrics.average_score * 100).toFixed(1)}%`);
    console.log('=====================================\n');
    
    res.json({
      status: 'success',
      evaluation,
      timestamp: new Date().toISOString()
    });
  } catch (err) {
    console.error('[Model Evaluation] Error:', err.message);
    res.status(500).json({ 
      error: 'Model evaluation failed',
      message: err.message 
    });
  }
});

// ─── SEARCH ───────────────────────────────────────────────────────────────────

app.post('/api/search', async (req, res) => {
  const { query, topK = 5 } = req.body;
  if (!query) return res.status(400).json({ error: 'query required' });

  const { data, error } = await supabase
    .rpc('search_medical', { query_text: query, match_count: topK });

  if (error) return res.status(500).json({ error: error.message });

  res.json({
    results: (data || []).map(r => ({
      title:   r.title,
      category: r.category,
      source:  r.source,
      snippet: r.content?.slice(0, 150),
    })),
  });
});

// ─── START ────────────────────────────────────────────────────────────────────

app.listen(PORT, async () => {
  console.log(`\n🏥 MedAssist RAG Backend v2.0`);
  console.log(`   http://localhost:${PORT}`);
  console.log(`   Mode: pgvector + MiniLM + Groq`);
  console.log(`   Model Evaluation: Available\n`);

  // Warm up embedding model in background (downloads once, ~25MB)
  console.log('[Startup] Warming up embedding model...');
  warmupEmbedder()
    .then(() => console.log('[Startup] Embedding model ready.'))
    .catch(e => console.warn('[Startup] Embedder warmup failed:', e.message));
});
