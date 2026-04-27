/**
 * embed.js
 * Generates sentence embeddings using all-MiniLM-L6-v2 via @xenova/transformers.
 *
 * - Model: sentence-transformers/all-MiniLM-L6-v2
 * - Dimensions: 384
 * - Runs locally — no API key, no cost
 * - Downloads model once (~25MB) and caches it
 */

import { pipeline, env } from '@xenova/transformers';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Use persistent disk on Render, local .cache otherwise
const CACHE_DIR = process.env.TRANSFORMERS_CACHE ||
  path.join(__dirname, '..', '.cache');

env.cacheDir = CACHE_DIR;
env.allowLocalModels = false;

let _embedder = null;

async function getEmbedder() {
  if (!_embedder) {
    console.log('[Embed] Loading all-MiniLM-L6-v2 (downloads once ~25MB)...');
    _embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    console.log('[Embed] Model ready.');
  }
  return _embedder;
}

/**
 * Embed a single text string.
 * Returns a Float32Array of 384 dimensions.
 *
 * @param {string} text
 * @returns {Promise<number[]>}
 */
export async function embedText(text) {
  const embedder = await getEmbedder();
  const output = await embedder(text, { pooling: 'mean', normalize: true });
  return Array.from(output.data);
}

/**
 * Embed a batch of texts.
 * Returns array of embedding arrays.
 *
 * @param {string[]} texts
 * @returns {Promise<number[][]>}
 */
export async function embedBatch(texts) {
  const embedder = await getEmbedder();
  const results = [];
  for (const text of texts) {
    const output = await embedder(text, { pooling: 'mean', normalize: true });
    results.push(Array.from(output.data));
  }
  return results;
}

/**
 * Warm up the model (call once at server start).
 */
export async function warmupEmbedder() {
  await embedText('warmup');
}
