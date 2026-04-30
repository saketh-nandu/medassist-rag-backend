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

// ─── STEP 3a: ENHANCED VECTOR RETRIEVAL ────────────────────────────────────────

async function vectorRetrieve(queryEmbedding, topK = 25) { // Increased for better recall
  if (!queryEmbedding) return [];

  // Try the vector function first
  try {
    const { data, error } = await supabase.rpc('match_chunks', {
      query_embedding: queryEmbedding,
      match_count: topK,
      similarity_threshold: 0.1
    });

    if (!error && data && data.length > 0) {
      console.log(`[RAG] Vector: ${data.length} chunks, top sim: ${data[0].similarity?.toFixed(3)}, avg: ${(data.reduce((sum, r) => sum + (r.similarity || 0), 0) / data.length).toFixed(3)}`);
      return data;
    }
  } catch (err) {
    console.warn('[RAG] Vector function failed:', err.message);
  }

  // Fallback to direct similarity calculation
  console.log('[RAG] Using direct similarity fallback...');
  return await directVectorSearch(queryEmbedding, topK);
}

async function directVectorSearch(queryEmbedding, topK) {
  // Get all chunks with embeddings
  const { data: chunks, error } = await supabase
    .from('medical_chunks')
    .select('id, doc_id, source, category, title, chunk_index, content, metadata, embedding')
    .not('embedding', 'is', null)
    .limit(1000); // Limit to prevent memory issues

  if (error || !chunks || chunks.length === 0) {
    console.warn('[RAG] Direct vector search failed:', error?.message || 'No chunks found');
    return await ftsFallback(queryEmbedding, topK);
  }

  // Calculate similarities
  const scored = chunks.map(c => {
    try {
      const emb = typeof c.embedding === 'string' ? JSON.parse(c.embedding) : c.embedding;
      if (!emb || !Array.isArray(emb) || emb.length !== queryEmbedding.length) {
        return { ...c, similarity: 0 };
      }
      return { ...c, similarity: cosineSim(queryEmbedding, emb) };
    } catch (err) {
      return { ...c, similarity: 0 };
    }
  });

  // Sort by similarity and return top results
  scored.sort((a, b) => b.similarity - a.similarity);
  const results = scored.slice(0, topK);
  
  console.log(`[RAG] Direct vector: ${results.length} chunks, top sim: ${results[0]?.similarity?.toFixed(3)}, avg: ${results.length > 0 ? (results.reduce((sum, r) => sum + r.similarity, 0) / results.length).toFixed(3) : 0}`);
  
  return results;
}

// ─── STEP 3b: ENHANCED KEYWORD (BM25-style) RETRIEVAL ─────────────────────────
// Uses Postgres full-text search as a second retrieval path with better medical term handling

async function keywordRetrieve(query, topK = 25) { // Increased for better recall
  // Enhanced medical term extraction with comprehensive medical vocabulary
  const stopwords = new Set(['i','me','my','the','a','an','is','are','was','were','have','has','do','does','can','could','would','should','what','how','why','when','where','which','this','that','these','those','and','or','but','for','with','about','from','to','of','in','on','at','by','as','it','its','be','been','being','am','will','shall','may','might','must','need','used','get','got','feel','feeling','having','been','very','so','just','also','more','some','any','all','no','not','than','then','there','here','up','down','out','off','over','under','again','further','once']);

  // Comprehensive medical synonym expansion
  const medicalExpansions = {
    'fever': ['fever', 'pyrexia', 'temperature', 'febrile', 'hyperthermia'],
    'headache': ['headache', 'cephalgia', 'migraine', 'head pain', 'cranial pain'],
    'pain': ['pain', 'ache', 'discomfort', 'soreness', 'tenderness', 'hurt'],
    'stomach': ['stomach', 'abdominal', 'gastric', 'belly', 'tummy', 'abdomen'],
    'heart': ['heart', 'cardiac', 'cardiovascular', 'coronary', 'myocardial'],
    'breathing': ['breathing', 'respiratory', 'dyspnea', 'breathless', 'respiration'],
    'diabetes': ['diabetes', 'diabetic', 'blood sugar', 'glucose', 'insulin'],
    'blood pressure': ['hypertension', 'hypotension', 'blood pressure', 'BP'],
    'infection': ['infection', 'bacterial', 'viral', 'fungal', 'pathogen', 'sepsis'],
    'allergy': ['allergy', 'allergic', 'hypersensitivity', 'anaphylaxis', 'reaction'],
    'cough': ['cough', 'coughing', 'bronchitis', 'respiratory', 'wheeze'],
    'rash': ['rash', 'skin', 'dermatitis', 'eczema', 'urticaria', 'hives'],
    'fatigue': ['fatigue', 'tired', 'weakness', 'exhaustion', 'lethargy'],
    'nausea': ['nausea', 'vomiting', 'emesis', 'queasiness', 'sick'],
    'diarrhea': ['diarrhea', 'loose stool', 'gastroenteritis', 'bowel'],
    'joint': ['joint', 'arthritis', 'rheumatoid', 'articular', 'synovial'],
    'throat': ['throat', 'pharyngitis', 'tonsillitis', 'laryngitis', 'sore throat']
  };

  // Extract and expand terms
  let expandedTerms = [];
  const originalTerms = query.toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(t => t.length > 2 && !stopwords.has(t));

  for (const term of originalTerms) {
    expandedTerms.push(term);
    
    // Add medical expansions
    for (const [key, expansions] of Object.entries(medicalExpansions)) {
      if (term.includes(key) || key.includes(term)) {
        expandedTerms.push(...expansions);
      }
    }
  }

  // Remove duplicates and limit terms
  expandedTerms = [...new Set(expandedTerms)].slice(0, 15);

  if (expandedTerms.length === 0) return [];

  // Multi-strategy keyword search with better error handling
  const strategies = [
    // Strategy 1: Simple OR matching for maximum recall
    expandedTerms.slice(0, 8).join(' | '),
    // Strategy 2: Individual term matching
    expandedTerms.slice(0, 6).join(' | '),
    // Strategy 3: Partial matching with wildcards
    expandedTerms.slice(0, 4).map(t => `${t}:*`).join(' | ')
  ];

  let allResults = [];
  const seenIds = new Set();

  for (let i = 0; i < strategies.length; i++) {
    const tsQuery = strategies[i];
    
    try {
      // Use ilike for simpler text matching as fallback
      const { data: textResults, error: textError } = await supabase
        .from('medical_chunks')
        .select('id, doc_id, source, category, title, chunk_index, content, metadata, embedding')
        .or(expandedTerms.slice(0, 5).map(term => `content.ilike.%${term}%`).join(','))
        .limit(Math.ceil(topK / 2));

      if (!textError && textResults) {
        for (let j = 0; j < textResults.length; j++) {
          const chunk = textResults[j];
          if (!seenIds.has(chunk.id)) {
            seenIds.add(chunk.id);
            
            // Enhanced scoring based on term frequency and medical relevance
            const content = chunk.content.toLowerCase();
            let termScore = 0;
            let medicalBonus = 0;
            
            for (const term of expandedTerms) {
              const termCount = (content.match(new RegExp(term, 'g')) || []).length;
              termScore += termCount;
              
              // Medical terms get bonus
              if (['fever', 'pain', 'headache', 'diabetes', 'heart', 'infection'].includes(term)) {
                medicalBonus += 0.2;
              }
            }
            
            const similarity = Math.min(1, (termScore * 0.1 + medicalBonus + (1 - j / textResults.length * 0.3)));
            
            allResults.push({ ...chunk, similarity });
          }
        }
      }

      // Also try full-text search if available
      try {
        const { data: ftsResults, error: ftsError } = await supabase
          .from('medical_chunks')
          .select('id, doc_id, source, category, title, chunk_index, content, metadata, embedding')
          .textSearch('content', tsQuery, { type: 'websearch', config: 'english' })
          .limit(Math.ceil(topK / 2));

        if (!ftsError && ftsResults) {
          for (let j = 0; j < ftsResults.length; j++) {
            const chunk = ftsResults[j];
            if (!seenIds.has(chunk.id)) {
              seenIds.add(chunk.id);
              const similarity = 1 - (j / ftsResults.length * 0.4); // FTS gets higher base score
              allResults.push({ ...chunk, similarity });
            }
          }
        }
      } catch (ftsErr) {
        // FTS might not be available, continue with text search results
      }

    } catch (error) {
      console.warn(`[RAG] Keyword strategy ${i + 1} failed:`, error.message);
    }
  }

  // Sort by similarity and return top results
  allResults.sort((a, b) => b.similarity - a.similarity);
  const finalResults = allResults.slice(0, topK);
  
  console.log(`[RAG] Keyword: ${finalResults.length} chunks, strategies used: ${strategies.length}, avg score: ${finalResults.length > 0 ? (finalResults.reduce((sum, r) => sum + r.similarity, 0) / finalResults.length).toFixed(3) : 0}`);
  
  return finalResults;
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

// ─── STEP 4: ENHANCED RECIPROCAL RANK FUSION (RRF) ───────────────────────────
// Advanced RRF with medical relevance weighting and source diversity

function reciprocalRankFusion(lists, k = 50) { // Reduced k for stronger ranking influence
  const scores = new Map(); // id → { score, chunk, sources }

  // Weight different retrieval methods
  const listWeights = [1.2, 1.0]; // Vector search gets slight preference

  for (let listIdx = 0; listIdx < lists.length; listIdx++) {
    const list = lists[listIdx];
    const weight = listWeights[listIdx] || 1.0;
    
    list.forEach((chunk, rank) => {
      const id = String(chunk.id || chunk.doc_id);
      const prev = scores.get(id) || { score: 0, chunk, sources: new Set() };
      
      // Enhanced RRF with medical category bonuses
      let categoryBonus = 0;
      if (chunk.category === 'disease') categoryBonus = 0.15;
      else if (chunk.category === 'qa') categoryBonus = 0.1;
      else if (chunk.category === 'emergency') categoryBonus = 0.2;
      
      // Source diversity bonus
      const sourceBonus = prev.sources.has(chunk.source) ? 0 : 0.05;
      prev.sources.add(chunk.source);
      
      // Calculate enhanced RRF score
      const rrfScore = weight / (k + rank + 1);
      const enhancedScore = rrfScore + categoryBonus + sourceBonus;
      
      scores.set(id, {
        score: prev.score + enhancedScore,
        chunk: { ...prev.chunk, ...chunk }, // merge fields, prefer later
        sources: prev.sources
      });
    });
  }

  return [...scores.values()]
    .sort((a, b) => b.score - a.score)
    .map(({ score, chunk }) => ({ ...chunk, rrfScore: score }));
}

// ─── STEP 5: ADVANCED CROSS-ENCODER RE-RANKING ───────────────────────────────
// Medical-focused cross-encoder with comprehensive term weighting and semantic analysis

function crossEncoderScore(query, chunkContent) {
  const queryLower = query.toLowerCase();
  const contentLower = chunkContent.toLowerCase();
  
  // Extract query terms with medical preprocessing
  const queryTerms = new Set(
    queryLower.replace(/[^a-z0-9\s]/g, ' ')
      .split(/\s+/)
      .filter(t => t.length > 2)
      .flatMap(term => {
        // Medical term expansion for better matching
        const expansions = {
          'fever': ['fever', 'pyrexia', 'temperature', 'febrile'],
          'pain': ['pain', 'ache', 'discomfort', 'hurt', 'sore'],
          'headache': ['headache', 'cephalgia', 'migraine'],
          'stomach': ['stomach', 'abdominal', 'gastric', 'belly'],
          'heart': ['heart', 'cardiac', 'cardiovascular'],
          'breathing': ['breathing', 'respiratory', 'dyspnea'],
          'diabetes': ['diabetes', 'diabetic', 'glucose', 'insulin'],
          'infection': ['infection', 'bacterial', 'viral', 'pathogen']
        };
        return expansions[term] || [term];
      })
  );
  
  const contentTerms = contentLower.replace(/[^a-z0-9\s]/g, ' ').split(/\s+/);

  // Advanced term matching with medical weighting
  let exactMatches = 0;
  let partialMatches = 0;
  let weightedScore = 0;
  let semanticBonus = 0;

  // Critical medical terms (highest priority)
  const criticalTerms = new Set([
    'emergency', 'critical', 'severe', 'acute', 'urgent', 'immediate',
    'heart attack', 'stroke', 'anaphylaxis', 'sepsis', 'shock', 'coma'
  ]);

  // High-priority medical terms
  const medicalTerms = new Set([
    'symptom', 'disease', 'treatment', 'diagnosis', 'medication', 'therapy',
    'syndrome', 'disorder', 'condition', 'infection', 'chronic', 'fever',
    'headache', 'nausea', 'vomiting', 'diarrhea', 'fatigue', 'weakness',
    'diabetes', 'hypertension', 'malaria', 'dengue', 'typhoid', 'pneumonia'
  ]);

  // Anatomical terms
  const anatomicalTerms = new Set([
    'head', 'brain', 'heart', 'lung', 'liver', 'kidney', 'stomach',
    'chest', 'abdomen', 'back', 'spine', 'joint', 'muscle', 'bone'
  ]);

  for (const contentTerm of contentTerms) {
    if (queryTerms.has(contentTerm)) {
      exactMatches++;
      
      // Apply medical term weighting
      if (criticalTerms.has(contentTerm)) {
        weightedScore += 5.0; // Critical medical terms
      } else if (medicalTerms.has(contentTerm)) {
        weightedScore += 3.0; // Important medical terms
      } else if (anatomicalTerms.has(contentTerm)) {
        weightedScore += 2.0; // Anatomical terms
      } else {
        weightedScore += 1.0; // Regular terms
      }
    } else {
      // Check for partial matches (medical synonyms)
      for (const queryTerm of queryTerms) {
        if (contentTerm.includes(queryTerm) || queryTerm.includes(contentTerm)) {
          partialMatches++;
          weightedScore += 0.5;
          break;
        }
      }
    }
  }

  // Semantic analysis bonuses
  const queryWords = Array.from(queryTerms);
  
  // Medical context bonus
  if (queryWords.some(w => medicalTerms.has(w)) && 
      contentTerms.some(w => medicalTerms.has(w))) {
    semanticBonus += 0.2;
  }
  
  // Symptom-disease correlation bonus
  const symptomWords = queryWords.filter(w => 
    ['fever', 'headache', 'pain', 'nausea', 'fatigue', 'cough', 'rash'].includes(w)
  );
  const diseaseWords = contentTerms.filter(w => 
    ['diabetes', 'malaria', 'dengue', 'typhoid', 'hypertension'].includes(w)
  );
  if (symptomWords.length > 0 && diseaseWords.length > 0) {
    semanticBonus += 0.15;
  }

  // Emergency context bonus
  if (queryWords.some(w => criticalTerms.has(w)) && 
      contentTerms.some(w => criticalTerms.has(w))) {
    semanticBonus += 0.3;
  }

  // Calculate final scores
  const coverage = queryTerms.size > 0 ? (exactMatches + partialMatches * 0.5) / queryTerms.size : 0;
  const density = contentTerms.length > 0 ? weightedScore / contentTerms.length : 0;
  const completeness = Math.min(1, exactMatches / Math.max(1, queryTerms.size));

  // Enhanced scoring formula with medical focus
  return Math.min(1, coverage * 0.4 + density * 0.3 + completeness * 0.2 + semanticBonus * 0.1);
}

function rerank(chunks, query, topK = 12) { // Increased for better selection
  const reranked = chunks
    .map(c => ({
      ...c,
      rerankScore: crossEncoderScore(query, c.content || ''),
    }))
    .sort((a, b) => {
      // Advanced blending: vector similarity + RRF + cross-encoder
      const vectorWeight = 0.35;
      const rrfWeight = 0.25;
      const rerankWeight = 0.4; // Increased cross-encoder influence
      
      const scoreA = (a.similarity || 0) * vectorWeight + 
                     (a.rrfScore || 0) * rrfWeight + 
                     a.rerankScore * rerankWeight;
      const scoreB = (b.similarity || 0) * vectorWeight + 
                     (b.rrfScore || 0) * rrfWeight + 
                     b.rerankScore * rerankWeight;
      
      return scoreB - scoreA;
    });

  return reranked.slice(0, topK);
}

// ─── STEP 6: ADVANCED MMR RE-RANKING ──────────────────────────────────────────
// Enhanced Maximal Marginal Relevance with medical diversity and quality focus

function mmrRerank(chunks, queryEmbedding, topK = 10, lambda = 0.75) { // Optimized parameters
  if (chunks.length <= topK) return chunks;
  if (!queryEmbedding) return chunks.slice(0, topK);

  const selected = [];
  const remaining = [...chunks];

  // Pre-calculate embeddings for efficiency
  const chunkEmbeddings = remaining.map(c => {
    const emb = typeof c.embedding === 'string' ? JSON.parse(c.embedding) : (c.embedding || []);
    return { chunk: c, embedding: emb };
  });

  while (selected.length < topK && remaining.length > 0) {
    let bestScore = -Infinity;
    let bestIdx = 0;

    for (let i = 0; i < remaining.length; i++) {
      const current = chunkEmbeddings[i];
      if (!current) continue;
      
      const chunk = current.chunk;
      const cEmb = current.embedding;

      // Base relevance score (combination of all previous scores)
      const vectorSim = chunk.similarity || 0;
      const rrfScore = chunk.rrfScore || 0;
      const rerankScore = chunk.rerankScore || 0;
      
      // Weighted relevance combining all signals
      const relevance = vectorSim * 0.4 + rrfScore * 0.3 + rerankScore * 0.3;

      // Calculate maximum redundancy with selected chunks
      let maxRedundancy = 0;
      for (const sel of selected) {
        const sEmb = typeof sel.embedding === 'string' ? JSON.parse(sel.embedding) : (sel.embedding || []);
        const sim = cosineSim(cEmb, sEmb);
        if (sim > maxRedundancy) maxRedundancy = sim;
      }

      // Enhanced diversity bonuses
      let diversityBonus = 0;
      
      // Category diversity bonus
      const selectedCategories = new Set(selected.map(s => s.category));
      if (!selectedCategories.has(chunk.category)) {
        diversityBonus += 0.15; // Bonus for new category
      }
      
      // Source diversity bonus
      const selectedSources = new Set(selected.map(s => s.source));
      if (!selectedSources.has(chunk.source)) {
        diversityBonus += 0.1; // Bonus for new source
      }
      
      // Medical priority bonus
      let medicalBonus = 0;
      if (chunk.category === 'emergency') medicalBonus += 0.2;
      else if (chunk.category === 'disease') medicalBonus += 0.15;
      else if (chunk.category === 'qa') medicalBonus += 0.1;
      
      // Quality bonus based on similarity scores
      const qualityBonus = Math.min(0.1, (vectorSim + rerankScore) * 0.05);

      // Enhanced MMR formula
      const mmrScore = lambda * relevance - 
                      (1 - lambda) * maxRedundancy + 
                      diversityBonus + 
                      medicalBonus + 
                      qualityBonus;

      if (mmrScore > bestScore) {
        bestScore = mmrScore;
        bestIdx = i;
      }
    }

    // Move best chunk from remaining to selected
    const bestChunk = remaining[bestIdx];
    selected.push(bestChunk);
    remaining.splice(bestIdx, 1);
    chunkEmbeddings.splice(bestIdx, 1);
  }

  return selected;
}

// ─── STEP 7: ENHANCED CONTEXT BUILDING (token-aware) ─────────────────────────

const MAX_CONTEXT_CHARS = 10000; // Increased for more comprehensive context

function buildContext(chunks) {
  // Advanced deduplication with medical priority and quality scoring
  const seen = new Map();
  
  for (const chunk of chunks) {
    const key = chunk.doc_id || chunk.title;
    const existing = seen.get(key);
    
    // Comprehensive priority calculation
    const vectorScore = chunk.similarity || 0;
    const rrfScore = chunk.rrfScore || 0;
    const rerankScore = chunk.rerankScore || 0;
    
    // Category priority weights
    const categoryWeight = {
      'emergency': 0.3,
      'disease': 0.2,
      'qa': 0.15,
      'treatment': 0.1,
      'prevention': 0.05
    }[chunk.category] || 0;
    
    // Source reliability weights
    const sourceWeight = {
      'MedQuAD': 0.15,
      'DSP': 0.1,
      'BuiltinKnowledge': 0.05
    }[chunk.source] || 0;
    
    const priority = vectorScore * 0.4 + 
                    rrfScore * 0.3 + 
                    rerankScore * 0.2 + 
                    categoryWeight + 
                    sourceWeight;
    
    if (!existing || priority > (existing.priority || 0)) {
      seen.set(key, { ...chunk, priority });
    }
  }

  const deduped = [...seen.values()].sort((a, b) => (b.priority || 0) - (a.priority || 0));
  const parts = [];
  let totalChars = 0;

  for (let i = 0; i < deduped.length; i++) {
    const c = deduped[i];
    
    // Enhanced metadata display
    const simPct = c.similarity !== undefined ? ` (${(c.similarity * 100).toFixed(0)}% match)` : '';
    const rrfInfo = c.rrfScore ? ` RRF:${c.rrfScore.toFixed(3)}` : '';
    const rerankInfo = c.rerankScore ? ` Rerank:${(c.rerankScore * 100).toFixed(0)}%` : '';
    const source = c.source ? ` | Source: ${c.source}` : '';
    const category = c.category ? ` | Category: ${c.category}` : '';
    
    const header = `[${i + 1}] ${c.title}${simPct}${rrfInfo}${rerankInfo}${source}${category}`;
    const budget = MAX_CONTEXT_CHARS - totalChars - header.length - 20; // Extra buffer
    
    if (budget < 200) break; // Increased minimum for better content
    
    // Smart content truncation - try to preserve complete sentences
    let content = c.content;
    if (content.length > budget) {
      content = content.slice(0, budget);
      const lastSentence = content.lastIndexOf('.');
      if (lastSentence > budget * 0.7) { // If we can preserve 70% and get complete sentence
        content = content.slice(0, lastSentence + 1);
      } else {
        content += '...';
      }
    }
    
    const entry = `${header}\n${content}`;
    parts.push(entry);
    totalChars += entry.length;
  }

  return parts.join('\n\n---\n\n');
}

// ─── STEP 8: ENHANCED LLM GENERATION WITH FALLBACK ──────────────────────────

async function generateWithGroq(userQuery, chunks, history = []) {
  const context = buildContext(chunks);

  const systemPrompt = `You are MedAssist AI, a highly knowledgeable and empathetic medical health assistant powered by an advanced RAG (Retrieval-Augmented Generation) system with comprehensive medical knowledge.

CRITICAL INSTRUCTIONS:
1. MEDICAL ACCURACY: Use ONLY the information in the provided medical knowledge context. Never hallucinate medical facts.
2. COMPREHENSIVE ANALYSIS: Think step-by-step through symptoms, possible conditions, and relevant treatments from the context.
3. STRUCTURED RESPONSE: Use clear markdown formatting:
   - **Brief Summary**: What the user is experiencing
   - **Possible Conditions**: List potential diagnoses with explanations
   - **Recommended Actions**: Specific, actionable medical advice
   - **When to Seek Care**: Clear guidance on urgency levels
   - **Precautions**: Important safety measures
4. MEDICAL TERMINOLOGY: Use appropriate medical terms but explain them clearly
5. EVIDENCE-BASED: Reference the medical knowledge provided and explain reasoning
6. SAFETY FIRST: Always err on the side of caution for medical advice

RESPONSE QUALITY STANDARDS:
- Be thorough and detailed - medical information deserves comprehensive answers
- Include specific symptoms, treatments, and precautions from the context
- Provide clear next steps and when to seek professional care
- Use empathetic, professional tone appropriate for healthcare
- Always end with the medical disclaimer

MANDATORY DISCLAIMER: "⚠️ This information is for educational purposes only. Please consult a qualified healthcare professional for proper diagnosis and treatment."`;

  // Enhanced history context with medical focus
  const historyMessages = history.slice(-4).map(h => ({
    role: h.role,
    content: h.content.slice(0, 400), // Slightly reduced for more context space
  }));

  const userPrompt = `## Medical Knowledge Context (Retrieved via Advanced RAG Pipeline)

${context}

---

## Patient Query
"${userQuery}"

Based on the comprehensive medical knowledge above, provide a thorough, structured, and medically accurate response. Analyze the symptoms carefully, consider all relevant conditions from the context, and provide detailed guidance while maintaining the highest medical standards.`;

  try {
    const chat = await groq.chat.completions.create({
      model: GROQ_MODEL,
      messages: [
        { role: 'system', content: systemPrompt },
        ...historyMessages,
        { role: 'user', content: userPrompt },
      ],
      temperature: 0.1,   // Lower temperature for more consistent medical responses
      max_tokens: 1500,   // Increased for more comprehensive responses
      top_p: 0.85,        // Slightly more focused
    });

    return chat.choices[0]?.message?.content?.trim() || null;
  } catch (error) {
    console.warn('[Groq] Generation failed, using enhanced fallback:', error.message);
    
    // Enhanced fallback generation using medical context
    return generateMedicalFallback(userQuery, chunks);
  }
}

// Enhanced fallback generation for when Groq fails
function generateMedicalFallback(userQuery, chunks) {
  if (!chunks || chunks.length === 0) {
    return `I apologize, but I couldn't find specific medical information for your query: "${userQuery}". Please consult a qualified healthcare professional for proper evaluation and guidance.

⚠️ This information is for educational purposes only. Please consult a qualified healthcare professional for proper diagnosis and treatment.`;
  }

  // Extract key medical information from top chunks
  const topChunks = chunks.slice(0, 3);
  const conditions = [...new Set(topChunks.map(c => c.title))];
  const categories = [...new Set(topChunks.map(c => c.category))];
  
  // Build comprehensive medical response
  let response = `## Medical Information for: "${userQuery}"\n\n`;
  
  // Add brief summary
  response += `**Brief Summary**: Based on your query, this may be related to ${conditions.slice(0, 2).join(' or ')}.\n\n`;
  
  // Add possible conditions
  response += `**Possible Conditions**:\n`;
  topChunks.forEach((chunk, i) => {
    const content = chunk.content || '';
    
    // Extract symptoms if available
    const symptomsMatch = content.match(/symptoms?:?\s*([^.]+)/i);
    const symptoms = symptomsMatch ? symptomsMatch[1].slice(0, 100) : 'Various symptoms may be present';
    
    // Extract treatment if available
    const treatmentMatch = content.match(/treatment:?\s*([^.]+)/i);
    const treatment = treatmentMatch ? treatmentMatch[1].slice(0, 100) : 'Consult healthcare provider for treatment';
    
    response += `${i + 1}. **${chunk.title}**: ${symptoms}. Treatment: ${treatment}.\n`;
  });
  
  response += `\n**Recommended Actions**:\n`;
  response += `- Monitor your symptoms carefully\n`;
  response += `- Keep a record of when symptoms started and their severity\n`;
  response += `- Stay hydrated and get adequate rest\n`;
  response += `- Avoid self-medication without professional guidance\n`;
  
  response += `\n**When to Seek Care**:\n`;
  if (categories.includes('emergency')) {
    response += `- **Seek immediate medical attention** - this may be an emergency condition\n`;
    response += `- Call emergency services (108) if symptoms are severe\n`;
  } else {
    response += `- Consult a healthcare provider if symptoms persist or worsen\n`;
    response += `- Seek immediate care if you develop severe symptoms\n`;
  }
  
  response += `\n**Precautions**:\n`;
  response += `- Follow proper hygiene practices\n`;
  response += `- Avoid contact with others if you have infectious symptoms\n`;
  response += `- Take prescribed medications as directed\n`;
  response += `- Don't ignore persistent or worsening symptoms\n`;
  
  response += `\n⚠️ This information is for educational purposes only. Please consult a qualified healthcare professional for proper diagnosis and treatment.`;
  
  return response;
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

  // ── STEP 3: Enhanced hybrid retrieval (vector + keyword in parallel) ──
  const [vectorChunks, keywordChunks] = await Promise.all([
    vectorRetrieve(queryEmbedding, 25), // Increased for better recall
    keywordRetrieve(userMessage, 25),   // Increased for better recall
  ]);

  // ── STEP 4: Enhanced Reciprocal Rank Fusion ──
  const fusedChunks = reciprocalRankFusion([vectorChunks, keywordChunks]);

  // ── STEP 5: Advanced cross-encoder re-ranking ──
  const reranked = rerank(fusedChunks, userMessage, 18); // Increased for better selection

  // ── STEP 6: Advanced MMR for diversity and quality ──
  const finalChunks = mmrRerank(reranked, queryEmbedding, 10, 0.75); // Optimized parameters

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
    console.log('[RAG] Using enhanced medical fallback response');
    content = generateMedicalFallback(userMessage, finalChunks);
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
