/**
 * rag.js — Enhanced Retrieval-Augmented Generation
 *
 * Pipeline:
 *   1. REWRITE   — expand + enrich query using symptom/medical ontology
 *   2. EMBED     — encode rewritten query → 384-dim vector (all-MiniLM-L6-v2)
 *   3. RETRIEVE  — hybrid: pgvector cosine + keyword BM25 scoring
 *   4. FUSE      — Reciprocal Rank Fusion (RRF) to merge ranked lists
 *   5. RERANK    — cross-encoder style scoring (query × chunk relevance)
 *   6. MMR       — Maximal Marginal Relevance for diversity
 *   7. GENERATE  — Groq llama-3.3-70b-versatile with chain-of-thought prompt
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

// Use the most capable free Groq model
const GROQ_MODEL = 'llama-3.3-70b-versatile';

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
    content: `I'm MedAssist AI — your AI-powered health assistant! 🏥\n\nI use a medical knowledge base with hybrid vector + keyword search to find the most relevant information for your question, then generate a response using a large language model.\n\nJust describe what you're experiencing and I'll help. Always consult a doctor for proper diagnosis.`,
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
// Expands the raw query with medical synonyms and related terms for better recall

const SYMPTOM_EXPANSION = {
  'headache':      'headache migraine tension cephalgia head pain intracranial pressure',
  'fever':         'fever pyrexia high temperature febrile hyperthermia chills',
  'cough':         'cough respiratory bronchitis cold productive dry cough mucus',
  'stomach':       'stomach abdominal pain gastric nausea gastritis peptic ulcer',
  'chest':         'chest pain cardiac heart angina respiratory pleurisy',
  'back':          'back pain spine lumbar musculoskeletal herniated disc sciatica',
  'rash':          'rash skin dermatitis eczema urticaria hives allergy eruption',
  'fatigue':       'fatigue tiredness weakness exhaustion lethargy chronic fatigue',
  'dizziness':     'dizziness vertigo balance vestibular lightheadedness syncope',
  'breathing':     'breathing respiratory shortness of breath dyspnea wheezing apnea',
  'joint':         'joint pain arthritis rheumatoid gout inflammation swelling',
  'throat':        'throat sore pharyngitis tonsillitis strep laryngitis',
  'diabetes':      'diabetes mellitus blood sugar insulin glucose hyperglycemia',
  'blood pressure':'hypertension hypotension blood pressure cardiovascular',
  'anxiety':       'anxiety panic disorder stress mental health nervousness',
  'depression':    'depression mood disorder mental health sadness hopelessness',
  'allergy':       'allergy allergic reaction hypersensitivity anaphylaxis',
  'infection':     'infection bacterial viral fungal pathogen sepsis',
  'pain':          'pain ache discomfort soreness tenderness inflammation',
  'swelling':      'swelling edema inflammation fluid retention bloating',
  'nausea':        'nausea vomiting emesis queasiness motion sickness',
  'diarrhea':      'diarrhea loose stool gastroenteritis bowel irritable',
  'insomnia':      'insomnia sleep disorder sleeplessness restless sleep apnea',
  'weight':        'weight loss gain obesity BMI metabolism thyroid',
};

function rewriteQuery(userQuery) {
  const lower = userQuery.toLowerCase();
  const expansions = new Set();

  for (const [kw, expansion] of Object.entries(SYMPTOM_EXPANSION)) {
    if (lower.includes(kw)) expansions.add(expansion);
  }

  if (expansions.size > 0) {
    return `${userQuery} ${[...expansions].join(' ')}`;
  }
  return userQuery;
}

// ─── STEP 2: EMBED QUERY ─────────────────────────────────────────────────────

async function embedQuery(query) {
  try {
    const rewritten = rewriteQuery(query);
    if (rewritten !== query) console.log(`[RAG] Query expanded: "${rewritten.slice(0, 100)}..."`);
    return await embedText(rewritten);
  } catch (err) {
    console.error('[RAG] Embedding failed:', err.message);
    return null;
  }
}

// ─── STEP 3a: VECTOR RETRIEVAL ────────────────────────────────────────────────

async function vectorRetrieve(queryEmbedding, topK = 15) {
  if (!queryEmbedding) return [];

  const { data, error } = await supabase.rpc('match_chunks', {
    query_embedding: queryEmbedding,
    match_count: topK,
  });

  if (!error && data && data.length > 0) {
    console.log(`[RAG] Vector: ${data.length} chunks, top sim: ${data[0].similarity?.toFixed(3)}`);
    return data;
  }

  if (error) console.warn('[RAG] Vector search error:', error.message);
  return await ftsFallback(queryEmbedding, topK);
}

// ─── STEP 3b: KEYWORD (BM25-style) RETRIEVAL ─────────────────────────────────
// Uses Postgres full-text search as a second retrieval path

async function keywordRetrieve(query, topK = 15) {
  // Extract meaningful medical terms (skip stopwords)
  const stopwords = new Set(['i','me','my','the','a','an','is','are','was','were','have','has','do','does','can','could','would','should','what','how','why','when','where','which','this','that','these','those','and','or','but','for','with','about','from','to','of','in','on','at','by','as','it','its','be','been','being','am','will','shall','may','might','must','need','used','get','got','feel','feeling','having','been','very','so','just','also','more','some','any','all','no','not','than','then','there','here','up','down','out','off','over','under','again','further','once']);

  const terms = query.toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(t => t.length > 2 && !stopwords.has(t));

  if (terms.length === 0) return [];

  // Build tsquery: term1 | term2 | term3
  const tsQuery = terms.slice(0, 8).join(' | ');

  const { data, error } = await supabase
    .from('medical_chunks')
    .select('id, doc_id, source, category, title, chunk_index, content, metadata, embedding')
    .textSearch('content', tsQuery, { type: 'websearch', config: 'english' })
    .limit(topK);

  if (error) {
    console.warn('[RAG] Keyword search error:', error.message);
    return [];
  }

  console.log(`[RAG] Keyword: ${(data || []).length} chunks`);
  return (data || []).map((c, i) => ({ ...c, similarity: 1 - (i / topK) * 0.5 }));
}

// ─── STEP 3c: FTS FALLBACK ────────────────────────────────────────────────────

async function ftsFallback(queryEmbedding, topK) {
  const { data: chunks } = await supabase
    .from('medical_chunks')
    .select('id, doc_id, source, category, title, chunk_index, content, metadata, embedding')
    .limit(200);

  if (chunks && chunks.length > 0 && chunks[0].embedding) {
    const scored = chunks.map(c => {
      const emb = typeof c.embedding === 'string' ? JSON.parse(c.embedding) : c.embedding;
      return { ...c, similarity: cosineSim(queryEmbedding, emb) };
    });
    scored.sort((a, b) => b.similarity - a.similarity);
    console.log(`[RAG] JS cosine fallback: top sim ${scored[0]?.similarity?.toFixed(3)}`);
    return scored.slice(0, topK);
  }

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

// ─── STEP 4: RECIPROCAL RANK FUSION (RRF) ────────────────────────────────────
// Merges two ranked lists (vector + keyword) into one unified ranking.
// RRF score = Σ 1 / (k + rank_i)  where k=60 is a smoothing constant.

function reciprocalRankFusion(lists, k = 60) {
  const scores = new Map(); // id → { score, chunk }

  for (const list of lists) {
    list.forEach((chunk, rank) => {
      const id = String(chunk.id || chunk.doc_id);
      const prev = scores.get(id) || { score: 0, chunk };
      scores.set(id, {
        score: prev.score + 1 / (k + rank + 1),
        chunk: { ...prev.chunk, ...chunk }, // merge fields
      });
    });
  }

  return [...scores.values()]
    .sort((a, b) => b.score - a.score)
    .map(({ score, chunk }) => ({ ...chunk, rrfScore: score }));
}

// ─── STEP 5: CROSS-ENCODER RE-RANKING ────────────────────────────────────────
// Lightweight lexical cross-encoder: scores each chunk by term overlap with query.
// A true neural cross-encoder would be ideal but this is fast and effective.

function crossEncoderScore(query, chunkContent) {
  const queryTerms = new Set(
    query.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter(t => t.length > 2)
  );
  const chunkTerms = chunkContent.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/);

  let matches = 0;
  let weightedMatches = 0;

  // Medical terms get higher weight
  const medicalBoost = new Set(['symptom','disease','treatment','diagnosis','medication','precaution','cause','prevention','therapy','syndrome','disorder','condition','infection','chronic','acute']);

  for (const term of chunkTerms) {
    if (queryTerms.has(term)) {
      matches++;
      weightedMatches += medicalBoost.has(term) ? 2 : 1;
    }
  }

  const coverage = queryTerms.size > 0 ? matches / queryTerms.size : 0;
  const density  = chunkTerms.length > 0 ? weightedMatches / chunkTerms.length : 0;

  return coverage * 0.7 + density * 0.3;
}

function rerank(chunks, query, topK = 8) {
  return chunks
    .map(c => ({
      ...c,
      rerankScore: crossEncoderScore(query, c.content || ''),
    }))
    .sort((a, b) => {
      // Blend: 60% vector similarity + 40% cross-encoder
      const scoreA = (a.similarity || a.rrfScore || 0) * 0.6 + a.rerankScore * 0.4;
      const scoreB = (b.similarity || b.rrfScore || 0) * 0.6 + b.rerankScore * 0.4;
      return scoreB - scoreA;
    })
    .slice(0, topK);
}

// ─── STEP 6: MMR RE-RANKING ───────────────────────────────────────────────────
// Maximal Marginal Relevance: balance relevance vs diversity

function mmrRerank(chunks, queryEmbedding, topK = 6, lambda = 0.65) {
  if (chunks.length <= topK) return chunks;
  if (!queryEmbedding) return chunks.slice(0, topK);

  const selected  = [];
  const remaining = [...chunks];

  while (selected.length < topK && remaining.length > 0) {
    let bestScore = -Infinity;
    let bestIdx   = 0;

    for (let i = 0; i < remaining.length; i++) {
      const c    = remaining[i];
      const cEmb = typeof c.embedding === 'string' ? JSON.parse(c.embedding) : (c.embedding || []);

      const relevance = c.similarity ?? cosineSim(queryEmbedding, cEmb);

      let maxRedundancy = 0;
      for (const sel of selected) {
        const sEmb = typeof sel.embedding === 'string' ? JSON.parse(sel.embedding) : (sel.embedding || []);
        const sim  = cosineSim(cEmb, sEmb);
        if (sim > maxRedundancy) maxRedundancy = sim;
      }

      const score = lambda * relevance - (1 - lambda) * maxRedundancy;
      if (score > bestScore) { bestScore = score; bestIdx = i; }
    }

    selected.push(remaining[bestIdx]);
    remaining.splice(bestIdx, 1);
  }

  return selected;
}

// ─── STEP 7: BUILD CONTEXT (token-aware) ─────────────────────────────────────

const MAX_CONTEXT_CHARS = 6000; // ~1500 tokens — safe for 70b model's 32k context

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
  const parts   = [];
  let totalChars = 0;

  for (let i = 0; i < deduped.length; i++) {
    const c       = deduped[i];
    const simPct  = c.similarity !== undefined ? ` (${(c.similarity * 100).toFixed(0)}% match)` : '';
    const source  = c.source ? ` | Source: ${c.source}` : '';
    const header  = `[${i + 1}] ${c.title}${simPct}${source}`;
    const budget  = MAX_CONTEXT_CHARS - totalChars - header.length - 10;
    if (budget < 100) break;
    const body  = c.content.slice(0, budget);
    const entry = `${header}\n${body}`;
    parts.push(entry);
    totalChars += entry.length;
  }

  return parts.join('\n\n---\n\n');
}

// ─── STEP 8: LLM GENERATION ──────────────────────────────────────────────────

async function generateWithGroq(userQuery, chunks, history = []) {
  const context = buildContext(chunks);

  const systemPrompt = `You are MedAssist AI, a highly knowledgeable and empathetic medical health assistant powered by a RAG (Retrieval-Augmented Generation) system.

INSTRUCTIONS:
1. THINK STEP BY STEP before answering. Internally reason about the symptoms, possible conditions, and relevant information from the context.
2. Use ONLY the information in the provided medical knowledge context. Do NOT hallucinate or invent facts.
3. Structure your response clearly using markdown:
   - Start with a brief summary of what the user is experiencing
   - List possible conditions or explanations
   - Provide actionable advice and precautions
   - Mention when to seek immediate medical attention
4. Be specific and detailed — the user deserves a thorough, helpful answer.
5. If the context is insufficient, say so honestly and suggest consulting a doctor.
6. ALWAYS end with: "⚠️ This information is for educational purposes only. Please consult a qualified healthcare professional for proper diagnosis and treatment."

TONE: Warm, clear, professional. Avoid jargon unless explained.`;

  // Include last 6 turns of conversation history for better context continuity
  const historyMessages = history.slice(-6).map(h => ({
    role: h.role,
    content: h.content.slice(0, 500),
  }));

  const userPrompt = `## Retrieved Medical Knowledge (via hybrid vector + keyword search)

${context}

---

## User Question
"${userQuery}"

Based on the retrieved medical knowledge above, provide a comprehensive, structured, and helpful response. Think through the symptoms and conditions carefully before answering.`;

  const chat = await groq.chat.completions.create({
    model: GROQ_MODEL,
    messages: [
      { role: 'system', content: systemPrompt },
      ...historyMessages,
      { role: 'user', content: userPrompt },
    ],
    temperature: 0.2,   // lower = more factual, less creative
    max_tokens: 1200,   // more room for detailed answers
    top_p: 0.9,
  });

  return chat.choices[0]?.message?.content?.trim() || null;
}

// ─── EXTRACT STRUCTURED METADATA FROM CHUNKS ─────────────────────────────────

function extractMetadata(chunks) {
  const conditions = [...new Set(
    chunks
      .filter(c => c.category === 'disease' || c.category === 'qa')
      .slice(0, 4)
      .map(c => c.title)
  )];

  const suggestedActions = [];
  for (const chunk of chunks) {
    const precMatch = chunk.content.match(/Precautions?:\s*(.+?)(?:\n|$)/i);
    if (precMatch) {
      const precs = precMatch[1].split(/[,;]/).map(p => p.trim()).filter(p => p.length > 5);
      suggestedActions.push(...precs.slice(0, 2));
    }
    if (suggestedActions.length >= 4) break;
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

  // ── STEP 1+2: Rewrite + Embed query ──
  const queryEmbedding = await embedQuery(userMessage);

  // ── STEP 3: Hybrid retrieval (vector + keyword in parallel) ──
  const [vectorChunks, keywordChunks] = await Promise.all([
    vectorRetrieve(queryEmbedding, 15),
    keywordRetrieve(userMessage, 15),
  ]);

  // ── STEP 4: Reciprocal Rank Fusion ──
  const fusedChunks = reciprocalRankFusion([vectorChunks, keywordChunks]);

  // ── STEP 5: Cross-encoder re-ranking ──
  const reranked = rerank(fusedChunks, userMessage, 12);

  // ── STEP 6: MMR for diversity ──
  const finalChunks = mmrRerank(reranked, queryEmbedding, 6, 0.65);

  if (!finalChunks || finalChunks.length === 0) {
    return {
      content: `I wasn't able to find relevant information for: "${userMessage}"\n\nPlease consult a qualified healthcare professional for proper evaluation.\n\n⚠️ *This is not a medical diagnosis.*`,
      conditions: ['Requires Medical Evaluation'],
      suggestedActions: ['Consult a doctor for proper diagnosis','Document when symptoms started','Stay hydrated and rest','Seek emergency care if symptoms worsen suddenly'],
      severity: 'low', affectedAreas: areas, isEmergency: false,
    };
  }

  // ── DETAILED RAG SCORING AND LOGGING WITH DATASET SOURCES ──
  console.log('\n🧠 ===== RAG PIPELINE DETAILED RESULTS =====');
  console.log(`📊 Pipeline Stats: ${vectorChunks.length} vector + ${keywordChunks.length} keyword → ${fusedChunks.length} fused → ${reranked.length} reranked → ${finalChunks.length} final`);
  
  // Function to get detailed dataset source information
  function getDatasetSource(chunk) {
    const metadata = chunk.metadata ? (typeof chunk.metadata === 'string' ? JSON.parse(chunk.metadata) : chunk.metadata) : {};
    let datasetInfo = '';
    
    if (chunk.source === 'MedQuAD') {
      const folder = metadata.folder || 'Unknown';
      const filename = metadata.filename || chunk.doc_id || 'Unknown';
      datasetInfo = `📁 MedQuAD/${folder}/${filename}`;
    } else if (chunk.source === 'DSP') {
      const csvFile = metadata.csv_file || metadata.source_file || 'dataset.csv';
      const rowIndex = metadata.row_index || metadata.index || 'Unknown';
      datasetInfo = `📊 DSP/${csvFile} (Row ${rowIndex})`;
    } else if (chunk.source === 'BuiltinKnowledge') {
      const knowledgeType = metadata.type || 'medical_knowledge';
      datasetInfo = `🏥 BuiltinKnowledge/${knowledgeType}`;
    } else {
      datasetInfo = `📄 ${chunk.source || 'Unknown'}`;
    }
    
    return datasetInfo;
  }
  
  // Log top vector results with detailed dataset sources
  console.log('\n🔍 TOP VECTOR SEARCH RESULTS:');
  vectorChunks.slice(0, 3).forEach((chunk, i) => {
    console.log(`  ${i + 1}. [${(chunk.similarity * 100).toFixed(1)}%] ${chunk.title?.slice(0, 50)}...`);
    console.log(`     📚 Source: ${chunk.source} | Category: ${chunk.category}`);
    console.log(`     ${getDatasetSource(chunk)}`);
    if (chunk.chunk_index !== undefined) console.log(`     📑 Chunk: ${chunk.chunk_index + 1} | Doc ID: ${chunk.doc_id}`);
  });
  
  // Log top keyword results with detailed dataset sources
  console.log('\n🔤 TOP KEYWORD SEARCH RESULTS:');
  keywordChunks.slice(0, 3).forEach((chunk, i) => {
    console.log(`  ${i + 1}. [${(chunk.similarity * 100).toFixed(1)}%] ${chunk.title?.slice(0, 50)}...`);
    console.log(`     📚 Source: ${chunk.source} | Category: ${chunk.category}`);
    console.log(`     ${getDatasetSource(chunk)}`);
    if (chunk.chunk_index !== undefined) console.log(`     📑 Chunk: ${chunk.chunk_index + 1} | Doc ID: ${chunk.doc_id}`);
  });
  
  // Log RRF scores with dataset sources
  console.log('\n🔄 RECIPROCAL RANK FUSION SCORES:');
  fusedChunks.slice(0, 3).forEach((chunk, i) => {
    console.log(`  ${i + 1}. [RRF: ${chunk.rrfScore?.toFixed(4)}] ${chunk.title?.slice(0, 50)}...`);
    console.log(`     ${getDatasetSource(chunk)}`);
  });
  
  // Log final re-ranked results with all scores and detailed dataset sources
  console.log('\n🎯 FINAL RE-RANKED RESULTS WITH DATASET SOURCES:');
  finalChunks.forEach((chunk, i) => {
    const vectorSim = chunk.similarity ? (chunk.similarity * 100).toFixed(1) : 'N/A';
    const rrfScore = chunk.rrfScore ? chunk.rrfScore.toFixed(4) : 'N/A';
    const rerankScore = chunk.rerankScore ? (chunk.rerankScore * 100).toFixed(1) : 'N/A';
    
    console.log(`  ${i + 1}. "${chunk.title?.slice(0, 45)}..."`);
    console.log(`     📈 Scores: Vector ${vectorSim}% | RRF ${rrfScore} | Rerank ${rerankScore}%`);
    console.log(`     📚 Source: ${chunk.source} | Category: ${chunk.category}`);
    console.log(`     ${getDatasetSource(chunk)}`);
    if (chunk.chunk_index !== undefined) console.log(`     📑 Chunk ${chunk.chunk_index + 1} of document "${chunk.doc_id}"`);
    console.log(`     📄 Content: ${chunk.content?.slice(0, 80)}...`);
    console.log('');
  });
  
  console.log('🤖 ===== LLM GENERATION PHASE =====');

  // ── STEP 7+8: Build context + Generate with Groq ──
  let content = null;
  let generationStats = {
    model: GROQ_MODEL,
    contextLength: 0,
    tokensUsed: 0,
    responseTime: 0
  };
  
  try {
    const startTime = Date.now();
    const context = buildContext(finalChunks);
    generationStats.contextLength = context.length;
    
    console.log(`📝 Context built: ${context.length} characters (~${Math.round(context.length / 4)} tokens)`);
    console.log(`🚀 Generating with ${GROQ_MODEL}...`);
    
    content = await generateWithGroq(userMessage, finalChunks, history);
    
    const endTime = Date.now();
    generationStats.responseTime = endTime - startTime;
    generationStats.tokensUsed = Math.round((context.length + (content?.length || 0)) / 4);
    
    console.log(`✅ Generation complete in ${generationStats.responseTime}ms`);
    console.log(`📊 Estimated tokens used: ${generationStats.tokensUsed}`);
    console.log(`📤 Response length: ${content?.length || 0} characters`);
    
  } catch (err) {
    console.error('❌ [Groq] Generation failed:', err.message);
    generationStats.error = err.message;
  }

  // Fallback if Groq fails
  if (!content) {
    const topChunk = finalChunks[0];
    content = `Based on your symptoms, this may be related to **${topChunk.title}**.\n\n${topChunk.content.slice(0, 500)}...\n\n⚠️ *This is not a medical diagnosis. Please consult a qualified healthcare professional.*`;
  }

  const { conditions, suggestedActions } = extractMetadata(finalChunks);

  // Prepare detailed response with RAG metadata
  const response = {
    content,
    conditions,
    suggestedActions,
    severity,
    affectedAreas: areas,
    isEmergency: false,
    // RAG Pipeline Metadata
    ragMetadata: {
      pipelineStats: {
        vectorResults: vectorChunks.length,
        keywordResults: keywordChunks.length,
        fusedResults: fusedChunks.length,
        rerankedResults: reranked.length,
        finalResults: finalChunks.length
      },
      topSources: finalChunks.map(chunk => {
        const metadata = chunk.metadata ? (typeof chunk.metadata === 'string' ? JSON.parse(chunk.metadata) : chunk.metadata) : {};
        let datasetPath = '';
        
        if (chunk.source === 'MedQuAD') {
          const folder = metadata.folder || 'Unknown';
          const filename = metadata.filename || chunk.doc_id || 'Unknown';
          datasetPath = `MedQuAD/${folder}/${filename}`;
        } else if (chunk.source === 'DSP') {
          const csvFile = metadata.csv_file || metadata.source_file || 'dataset.csv';
          const rowIndex = metadata.row_index || metadata.index || 'Unknown';
          datasetPath = `DSP/${csvFile} (Row ${rowIndex})`;
        } else if (chunk.source === 'BuiltinKnowledge') {
          const knowledgeType = metadata.type || 'medical_knowledge';
          datasetPath = `BuiltinKnowledge/${knowledgeType}`;
        } else {
          datasetPath = chunk.source || 'Unknown';
        }
        
        return {
          title: chunk.title,
          source: chunk.source,
          category: chunk.category,
          datasetPath: datasetPath,
          chunkIndex: chunk.chunk_index,
          docId: chunk.doc_id,
          vectorScore: chunk.similarity ? (chunk.similarity * 100).toFixed(1) : null,
          rrfScore: chunk.rrfScore ? chunk.rrfScore.toFixed(4) : null,
          rerankScore: chunk.rerankScore ? (chunk.rerankScore * 100).toFixed(1) : null,
          metadata: metadata
        };
      }),
      generationStats,
      queryExpansion: rewriteQuery(userMessage) !== userMessage ? rewriteQuery(userMessage) : null
    }
  };
  
  console.log('\n🎉 ===== RAG PIPELINE COMPLETE =====');
  console.log(`📋 Final Response: ${content?.slice(0, 100)}...`);
  console.log(`🏥 Conditions: ${conditions.join(', ')}`);
  console.log(`⚠️  Severity: ${severity}`);
  console.log('=======================================\n');

  return response;
}
