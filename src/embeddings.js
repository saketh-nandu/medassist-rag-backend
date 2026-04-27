/**
 * embeddings.js
 * Handles text → vector embedding
 * Supports OpenAI (1536-dim) with a simple fallback
 */

import OpenAI from 'openai';
import dotenv from 'dotenv';
dotenv.config();

let openai = null;
if (process.env.OPENAI_API_KEY) {
  openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
}

const EMBEDDING_MODEL = 'text-embedding-3-small'; // 1536 dims, cheap ($0.02/1M tokens)
const BATCH_SIZE = 100; // OpenAI allows up to 2048 inputs per request

/**
 * Embed a single text string → float[]
 */
export async function embedText(text) {
  if (!openai) throw new Error('OPENAI_API_KEY not set. Cannot generate embeddings.');

  const cleaned = text.replace(/\n+/g, ' ').trim().slice(0, 8000);
  const res = await openai.embeddings.create({
    model: EMBEDDING_MODEL,
    input: cleaned,
  });
  return res.data[0].embedding;
}

/**
 * Embed a batch of texts → float[][]
 * Automatically batches to avoid rate limits
 */
export async function embedBatch(texts) {
  if (!openai) throw new Error('OPENAI_API_KEY not set.');

  const results = [];
  for (let i = 0; i < texts.length; i += BATCH_SIZE) {
    const batch = texts.slice(i, i + BATCH_SIZE).map((t) =>
      t.replace(/\n+/g, ' ').trim().slice(0, 8000)
    );
    const res = await openai.embeddings.create({
      model: EMBEDDING_MODEL,
      input: batch,
    });
    results.push(...res.data.map((d) => d.embedding));

    // Rate limit: ~3000 RPM on free tier
    if (i + BATCH_SIZE < texts.length) {
      await new Promise((r) => setTimeout(r, 200));
    }
    console.log(`  Embedded ${Math.min(i + BATCH_SIZE, texts.length)}/${texts.length}`);
  }
  return results;
}
