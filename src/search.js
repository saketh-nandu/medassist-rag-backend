/**
 * search.js
 * TF-IDF based search over the in-memory knowledge base.
 * No embeddings, no API calls — pure JavaScript.
 */

import { knowledgeBase } from './knowledge.js';

// ─── STOP WORDS ───────────────────────────────────────────────────────────────

const STOP_WORDS = new Set([
  'i', 'me', 'my', 'have', 'has', 'had', 'been', 'am', 'is', 'are', 'was',
  'were', 'be', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
  'may', 'might', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
  'to', 'for', 'of', 'with', 'by', 'from', 'as', 'into', 'through', 'about',
  'this', 'that', 'these', 'those', 'it', 'its', 'what', 'which', 'who',
  'when', 'where', 'why', 'how', 'all', 'some', 'any', 'no', 'not', 'very',
  'just', 'also', 'so', 'if', 'then', 'than', 'too', 'can', 'get', 'got',
  'feel', 'feeling', 'having', 'getting', 'since', 'past', 'last', 'days',
  'day', 'week', 'weeks', 'month', 'months', 'year', 'years', 'ago', 'now',
  'still', 'already', 'again', 'back', 'up', 'down', 'out', 'over', 'under',
  'more', 'most', 'much', 'many', 'few', 'little', 'lot', 'really', 'quite',
]);

// ─── TOKENIZE ─────────────────────────────────────────────────────────────────

function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(w => w.length > 2 && !STOP_WORDS.has(w));
}

// ─── SCORE DISEASE MATCH ──────────────────────────────────────────────────────

function scoreDisease(disease, queryTokens, queryRaw) {
  let score = 0;
  const queryLower = queryRaw.toLowerCase();

  // Exact disease name match — highest weight
  if (queryLower.includes(disease.nameLower)) {
    score += 20;
  }

  // Partial name match
  const nameTokens = tokenize(disease.name);
  nameTokens.forEach(nt => {
    if (queryTokens.includes(nt)) score += 5;
  });

  // Symptom matches
  disease.symptoms.forEach(symptom => {
    const symTokens = tokenize(symptom);
    symTokens.forEach(st => {
      if (queryTokens.includes(st)) {
        // Weight by symptom severity if available
        const sevWeight = knowledgeBase.symptomSeverity[symptom] || 1;
        score += 2 + Math.min(sevWeight, 5);
      }
    });
    // Exact symptom phrase match
    if (queryLower.includes(symptom.toLowerCase())) {
      score += 4;
    }
  });

  // Description keyword match
  if (disease.description) {
    const descTokens = tokenize(disease.description);
    queryTokens.forEach(qt => {
      if (descTokens.includes(qt)) score += 0.5;
    });
  }

  return score;
}

// ─── SCORE QA MATCH ──────────────────────────────────────────────────────────

function scoreQA(qa, queryTokens, queryRaw) {
  let score = 0;
  const queryLower = queryRaw.toLowerCase();

  // Focus/topic match
  if (queryLower.includes(qa.focusLower)) score += 15;
  const focusTokens = tokenize(qa.focus);
  focusTokens.forEach(ft => {
    if (queryTokens.includes(ft)) score += 4;
  });

  // Question similarity
  const qTokens = tokenize(qa.question);
  let qMatches = 0;
  queryTokens.forEach(qt => {
    if (qTokens.includes(qt)) qMatches++;
  });
  score += qMatches * 2;

  // Boost by question type relevance
  if (qa.qtype === 'symptoms' && queryLower.match(/symptom|sign|feel|experience/)) score += 3;
  if (qa.qtype === 'treatment' && queryLower.match(/treat|cure|medicine|medication|help/)) score += 3;
  if (qa.qtype === 'causes' && queryLower.match(/cause|why|reason|risk/)) score += 3;

  return score;
}

// ─── MAIN SEARCH ─────────────────────────────────────────────────────────────

export function search(query, topK = 5) {
  const queryTokens = tokenize(query);

  // Score all diseases
  const diseaseScores = knowledgeBase.diseases
    .map(d => ({ item: d, score: scoreDisease(d, queryTokens, query), type: 'disease' }))
    .filter(x => x.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);

  // Score QA pairs (only if we have query tokens to match)
  const qaScores = queryTokens.length > 0
    ? knowledgeBase.qa
        .map(q => ({ item: q, score: scoreQA(q, queryTokens, query), type: 'qa' }))
        .filter(x => x.score > 3)
        .sort((a, b) => b.score - a.score)
        .slice(0, topK)
    : [];

  // Merge: prioritize diseases, fill rest with QA
  const results = [];
  const seen = new Set();

  diseaseScores.forEach(r => {
    if (!seen.has(r.item.name)) {
      results.push(r);
      seen.add(r.item.name);
    }
  });

  qaScores.forEach(r => {
    if (results.length < topK * 2 && !seen.has(r.item.question)) {
      results.push(r);
      seen.add(r.item.question);
    }
  });

  return results.slice(0, topK * 2);
}

// ─── EXTRACT SYMPTOMS FROM QUERY ─────────────────────────────────────────────

export function extractSymptoms(query) {
  const queryLower = query.toLowerCase();
  const found = [];

  // Check against all known symptoms in KB
  const allSymptoms = new Set();
  knowledgeBase.diseases.forEach(d => d.symptoms.forEach(s => allSymptoms.add(s)));

  allSymptoms.forEach(sym => {
    if (queryLower.includes(sym.toLowerCase())) {
      found.push(sym);
    }
  });

  // Also extract tokens as potential symptoms
  const tokens = tokenize(query);
  tokens.forEach(t => {
    if (t.length > 4 && !found.includes(t)) found.push(t);
  });

  return [...new Set(found)].slice(0, 10);
}
