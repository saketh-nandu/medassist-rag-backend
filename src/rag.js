/**
 * rag.js — ACTUAL Retrieval-Augmented Generation
 *
 * Pipeline:
 *   1. EMBED   — encode user query → 384-dim vector (all-MiniLM-L6-v2)
 *   2. RETRIEVE — cosine similarity search in Supabase pgvector (medical_chunks)
 *   3. AUGMENT  — build context string from top-k chunks
 *   4. GENERATE — Groq LLM (llama-3.1-8b-instant) produces grounded answer
 */

import { createClient } from '@supabase/supabase-js';
import Groq from 'groq-sdk';
import dotenv from 'dotenv';
import { embedText } from './embed.js';
dotenv.config();

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY
);

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
const GROQ_MODEL = 'llama-3.1-8b-instant';

// ─── GREETING / SMALL-TALK DETECTION ─────────────────────────────────────────

const GREETINGS  = ['hi','hello','hey','hii','helo','howdy','good morning','good afternoon','good evening','namaste'];
const SMALL_TALK = ['how are you','what can you do','who are you','what are you'];
const THANKS     = ['thank you','thanks','ok thanks','okay thanks','great thanks'];

function isGreeting(text) {
  const lower = text.toLowerCase().trim();
  if (GREETINGS.some(g => lower === g || lower.startsWith(g + ' ') || lower.startsWith(g + '!'))) return 'greeting';
  if (SMALL_TALK.some(s => lower.includes(s))) return 'smalltalk';
  if (THANKS.some(t => lower.includes(t))) return 'thanks';
  return null;
}

function greetingResponse(type) {
  const hour = new Date().getHours();
  const timeGreeting = hour < 12 ? 'Good morning' : hour < 17 ? 'Good afternoon' : 'Good evening';
  if (type === 'greeting') return {
    content: `${timeGreeting}! 👋 I'm MedAssist AI, your personal health assistant.\n\nI can help you with:\n• **Symptom analysis** — describe what you're feeling\n• **Disease information** — ask about any condition\n• **Medication guidance** — dosage and precautions\n• **First aid advice** — for injuries and emergencies\n\nHow are you feeling today?`,
    conditions: [], suggestedActions: ['Describe your symptoms to get started','Ask about a specific disease','Ask about medications'],
    severity: 'low', affectedAreas: [], isEmergency: false,
  };
  if (type === 'smalltalk') return {
    content: `I'm MedAssist AI — your AI-powered health assistant! 🏥\n\nI use a medical knowledge base with vector search to find the most relevant information for your question, then generate a response using an LLM.\n\nJust describe what you're experiencing and I'll help. Always consult a doctor for proper diagnosis.`,
    conditions: [], suggestedActions: ['Tell me your symptoms','Ask about a health condition'],
    severity: 'low', affectedAreas: [], isEmergency: false,
  };
  return {
    content: `You're welcome! 😊 Take care of yourself. Feel free to ask me anything else about your health!`,
    conditions: [], suggestedActions: ['Stay hydrated','Get adequate rest','Consult a doctor if symptoms persist'],
    severity: 'low', affectedAreas: [], isEmergency: false,
  };
}

// ─── EMERGENCY DETECTION ─────────────────────────────────────────────────────

const EMERGENCY_PATTERNS = [
  'chest pain','chest pressure','heart attack',"can't breathe",'cannot breathe',
  'difficulty breathing','not breathing','stroke','unconscious','unresponsive',
  'seizure','severe bleeding','coughing blood','vomiting blood','sudden numbness',
  'face drooping','arm weakness','sudden severe headache','loss of consciousness',
  'choking','anaphylaxis','severe allergic reaction','overdose','poisoning',
  'collapsed','fainted','suicidal',
];

function isEmergency(text) {
  return EMERGENCY_PATTERNS.some(p => text.toLowerCase().includes(p));
}

// ─── BODY AREA DETECTION ─────────────────────────────────────────────────────

const AREA_MAP = {
  head:     ['headache','head','migraine','dizziness','vertigo','brain','eye','ear','nose','face','forehead','temple','scalp'],
  throat:   ['throat','neck','swallowing','tonsil','voice','hoarse','strep'],
  chest:    ['chest','heart','lung','breathing','breath','palpitation','cough','wheeze','asthma','rib'],
  abdomen:  ['stomach','abdomen','belly','nausea','vomit','diarrhea','constipation','bowel','intestine','liver','appendix','kidney','bloating','gastric'],
  back:     ['back','spine','lumbar','sciatica','vertebra','disc','lower back'],
  leftArm:  ['left arm','left shoulder','left elbow','left wrist','left hand'],
  rightArm: ['right arm','right shoulder','right elbow','right wrist','right hand','arm pain','shoulder pain'],
  leftLeg:  ['left leg','left knee','left ankle','left foot','left hip'],
  rightLeg: ['right leg','right knee','right ankle','right foot','right hip','leg pain','knee pain'],
};

function detectAreas(text) {
  const lower = text.toLowerCase();
  return Object.entries(AREA_MAP).filter(([, kws]) => kws.some(kw => lower.includes(kw))).map(([area]) => area);
}

// ─── SEVERITY ────────────────────────────────────────────────────────────────

function calcSeverity(text) {
  const lower = text.toLowerCase();
  if (['severe','extreme','unbearable','worst','crushing','sudden','acute','critical','excruciating'].some(w => lower.includes(w))) return 'high';
  if (['moderate','persistent','recurring','worsening','high fever','cannot sleep','getting worse'].some(w => lower.includes(w))) return 'medium';
  return 'low';
}

// ─── STEP 1: QUERY REWRITING ─────────────────────────────────────────────────
// Expand the raw user query into a richer search query for better retrieval

function rewriteQuery(userQuery) {
  const lower = userQuery.toLowerCase();
  const expansions = [];

  // Symptom → disease expansion
  const symptomMap = {
    'headache': 'headache migraine tension head pain',
    'fever': 'fever temperature pyrexia high temperature',
    'cough': 'cough respiratory bronchitis cold',
    'stomach': 'stomach abdominal pain gastric nausea',
    'chest': 'chest pain cardiac heart respiratory',
    'back': 'back pain spine lumbar musculoskeletal',
    'rash': 'rash skin dermatitis allergy',
    'fatigue': 'fatigue tiredness weakness exhaustion',
    'dizziness': 'dizziness vertigo balance head',
    'breathing': 'breathing respiratory shortness of breath dyspnea',
  };

  for (const [kw, expansion] of Object.entries(symptomMap)) {
    if (lower.includes(kw)) expansions.push(expansion);
  }

  // If we found expansions, append them
  if (expansions.length > 0) {
    return `${userQuery} ${expansions.join(' ')}`;
  }
  return userQuery;
}

// ─── STEP 1: EMBED QUERY ─────────────────────────────────────────────────────

async function embedQuery(query) {
  try {
    const rewritten = rewriteQuery(query);
    if (rewritten !== query) console.log(`[RAG] Query rewritten: "${rewritten.slice(0, 80)}..."`);
    return await embedText(rewritten);
  } catch (err) {
    console.error('[RAG] Embedding failed:', err.message);
    return null;
  }
}

// ─── STEP 2: VECTOR RETRIEVAL ─────────────────────────────────────────────────

async function retrieveChunks(queryEmbedding, topK = 8) {
  if (!queryEmbedding) return [];

  // Primary: vector similarity search via pgvector
  const { data, error } = await supabase.rpc('match_chunks', {
    query_embedding: queryEmbedding,  // pass as array — supabase-js handles serialization
    match_count: topK,
  });

  if (!error && data && data.length > 0) {
    console.log(`[RAG] Vector search: ${data.length} chunks, top similarity: ${data[0].similarity?.toFixed(3)}`);
    return data;
  }

  if (error) console.warn('[RAG] Vector search error:', error.message);

  // Fallback: FTS if match_chunks function not yet created
  console.warn('[RAG] Falling back to FTS — run SQL in setup-vector.sql to enable vector search');
  return await ftsFallback(queryEmbedding, topK);
}

async function ftsFallback(queryEmbedding, topK) {
  // Try medical_chunks with manual cosine (slow but works without the function)
  const { data: chunks } = await supabase
    .from('medical_chunks')
    .select('id, doc_id, source, category, title, chunk_index, content, metadata, embedding')
    .limit(200);

  if (chunks && chunks.length > 0 && chunks[0].embedding) {
    // Manual cosine similarity in JS
    const scored = chunks.map(c => {
      const emb = typeof c.embedding === 'string' ? JSON.parse(c.embedding) : c.embedding;
      const sim = cosineSim(queryEmbedding, emb);
      return { ...c, similarity: sim };
    });
    scored.sort((a, b) => b.similarity - a.similarity);
    console.log(`[RAG] JS cosine fallback: top similarity ${scored[0]?.similarity?.toFixed(3)}`);
    return scored.slice(0, topK);
  }

  // Last resort: FTS on old table
  const { data } = await supabase
    .from('medical_knowledge')
    .select('id, source, category, title, content, symptoms, precautions, metadata')
    .limit(topK);
  return (data || []).map(r => ({ ...r, chunk_index: 0, doc_id: r.title, similarity: 0 }));
}

function cosineSim(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot   += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-10);
}

// ─── STEP 3: MMR RE-RANKING ──────────────────────────────────────────────────
// Maximal Marginal Relevance: balance relevance vs diversity
// Avoids returning 5 chunks all saying the same thing

function mmrRerank(chunks, queryEmbedding, topK = 5, lambda = 0.6) {
  if (chunks.length <= topK) return chunks;
  if (!queryEmbedding) return chunks.slice(0, topK);

  const selected = [];
  const remaining = [...chunks];

  while (selected.length < topK && remaining.length > 0) {
    let bestScore = -Infinity;
    let bestIdx   = 0;

    for (let i = 0; i < remaining.length; i++) {
      const c = remaining[i];
      const cEmb = typeof c.embedding === 'string' ? JSON.parse(c.embedding) : (c.embedding || []);

      // Relevance to query
      const relevance = c.similarity ?? cosineSim(queryEmbedding, cEmb);

      // Max similarity to already-selected chunks (redundancy penalty)
      let maxRedundancy = 0;
      for (const sel of selected) {
        const sEmb = typeof sel.embedding === 'string' ? JSON.parse(sel.embedding) : (sel.embedding || []);
        const sim = cosineSim(cEmb, sEmb);
        if (sim > maxRedundancy) maxRedundancy = sim;
      }

      // MMR score: λ * relevance - (1-λ) * redundancy
      const score = lambda * relevance - (1 - lambda) * maxRedundancy;
      if (score > bestScore) { bestScore = score; bestIdx = i; }
    }

    selected.push(remaining[bestIdx]);
    remaining.splice(bestIdx, 1);
  }

  return selected;
}

// ─── STEP 3: BUILD CONTEXT (token-aware) ─────────────────────────────────────

const MAX_CONTEXT_CHARS = 3000; // ~750 tokens — safe for Groq 8k context

function buildContext(chunks) {
  // Deduplicate by doc_id — keep highest similarity chunk per document
  const seen = new Map();
  for (const chunk of chunks) {
    const key = chunk.doc_id || chunk.title;
    if (!seen.has(key) || (chunk.similarity ?? 0) > (seen.get(key).similarity ?? 0)) {
      seen.set(key, chunk);
    }
  }

  const deduped = [...seen.values()];
  const parts = [];
  let totalChars = 0;

  for (let i = 0; i < deduped.length; i++) {
    const c = deduped[i];
    const simPct = c.similarity !== undefined ? ` (${(c.similarity * 100).toFixed(0)}% match)` : '';
    const header = `[${i + 1}] ${c.title}${simPct}`;
    // Truncate chunk content to fit budget
    const remaining = MAX_CONTEXT_CHARS - totalChars - header.length - 10;
    if (remaining < 100) break;
    const body = c.content.slice(0, remaining);
    const entry = `${header}\n${body}`;
    parts.push(entry);
    totalChars += entry.length;
  }

  return parts.join('\n\n---\n\n');
}

// ─── STEP 4: LLM GENERATION (with conversation history) ──────────────────────

async function generateWithGroq(userQuery, chunks, history = []) {
  const context = buildContext(chunks);

  const systemPrompt = `You are MedAssist AI, a knowledgeable and empathetic medical health assistant.
You are given retrieved medical knowledge chunks found via semantic vector search (RAG).
Use ONLY the information in the provided context to answer the user's question.
Be clear, concise, and structured. Use markdown formatting (bold, bullet points).
Always end with a disclaimer reminding the user to consult a qualified healthcare professional.
Do NOT make up information not present in the context.
If the context does not contain enough information, say so honestly.`;

  // Build messages with conversation history (last 4 turns max)
  const historyMessages = history.slice(-4).map(h => ({
    role: h.role,
    content: h.content.slice(0, 300), // truncate old messages
  }));

  const userPrompt = `Retrieved medical knowledge (via vector similarity search):
${context}

User question: "${userQuery}"

Based on the retrieved context above, provide a helpful, structured medical response.`;

  const chat = await groq.chat.completions.create({
    model: GROQ_MODEL,
    messages: [
      { role: 'system', content: systemPrompt },
      ...historyMessages,
      { role: 'user', content: userPrompt },
    ],
    temperature: 0.3,
    max_tokens: 700,
  });

  return chat.choices[0]?.message?.content?.trim() || null;
}

// ─── EXTRACT STRUCTURED METADATA FROM CHUNKS ─────────────────────────────────

function extractMetadata(chunks) {
  const conditions = [...new Set(
    chunks
      .filter(c => c.category === 'disease')
      .slice(0, 3)
      .map(c => c.title)
  )];

  // Extract precautions from chunk content
  const suggestedActions = [];
  for (const chunk of chunks) {
    const precMatch = chunk.content.match(/Precautions?:\s*(.+?)(?:\n|$)/i);
    if (precMatch) {
      const precs = precMatch[1].split(/[,;]/).map(p => p.trim()).filter(p => p.length > 5);
      suggestedActions.push(...precs.slice(0, 2));
    }
    if (suggestedActions.length >= 3) break;
  }
  suggestedActions.push('Consult a doctor for proper diagnosis');
  suggestedActions.push('Monitor and document your symptoms');

  return {
    conditions,
    suggestedActions: [...new Set(suggestedActions)].slice(0, 5),
  };
}

// ─── MAIN RAG PIPELINE ───────────────────────────────────────────────────────

export async function ragQuery(userMessage, history = []) {
  // Handle greetings without RAG
  const greetType = isGreeting(userMessage);
  if (greetType) return greetingResponse(greetType);

  const areas    = detectAreas(userMessage);
  const severity = isEmergency(userMessage) ? 'high' : calcSeverity(userMessage);

  // Emergency — skip RAG, return immediately
  if (severity === 'high' && isEmergency(userMessage)) {
    return {
      content: '🚨 **EMERGENCY DETECTED**\n\nYour symptoms suggest a possible medical emergency. Please call **108** immediately.\n\n• Do NOT drive yourself to the hospital\n• Sit or lie down in a comfortable position\n• Loosen tight clothing\n• Stay on the line with emergency services\n• Keep someone with you at all times',
      conditions: ['Possible Medical Emergency'],
      suggestedActions: ['CALL 108 IMMEDIATELY','Do not drive yourself','Sit or lie down','Loosen tight clothing','Stay on line with emergency services'],
      severity: 'high',
      affectedAreas: areas.length > 0 ? areas : ['chest','head'],
      isEmergency: true,
    };
  }

  // ── STEP 1: Rewrite + Embed query ──
  const queryEmbedding = await embedQuery(userMessage);

  // ── STEP 2: Retrieve top-k chunks via vector search ──
  const rawChunks = await retrieveChunks(queryEmbedding, 12); // fetch more for re-ranking

  // ── STEP 2b: MMR re-rank for diversity ──
  const chunks = mmrRerank(rawChunks, queryEmbedding, 6, 0.65);

  if (!chunks || chunks.length === 0) {
    return {
      content: `I wasn't able to find relevant information for: "${userMessage}"\n\nPlease consult a qualified healthcare professional for proper evaluation.\n\n⚠️ *This is not a medical diagnosis.*`,
      conditions: ['Requires Medical Evaluation'],
      suggestedActions: ['Consult a doctor for proper diagnosis','Document when symptoms started','Stay hydrated and rest','Seek emergency care if symptoms worsen suddenly'],
      severity: 'low', affectedAreas: areas, isEmergency: false,
    };
  }

  // ── STEP 3 & 4: Build context + Generate with Groq (+ history) ──
  let content = null;
  try {
    content = await generateWithGroq(userMessage, chunks, history);
  } catch (err) {
    console.error('[Groq] Generation failed:', err.message);
  }

  // Fallback if Groq fails
  if (!content) {
    const topChunk = chunks[0];
    content = `Based on your symptoms, this may be related to **${topChunk.title}**.\n\n${topChunk.content.slice(0, 400)}...\n\n⚠️ *This is not a medical diagnosis. Please consult a qualified healthcare professional.*`;
  }

  const { conditions, suggestedActions } = extractMetadata(chunks);

  return {
    content,
    conditions,
    suggestedActions,
    severity,
    affectedAreas: areas,
    isEmergency: false,
  };
}
