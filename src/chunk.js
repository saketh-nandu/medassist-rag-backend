/**
 * chunk.js
 * Splits document text into overlapping chunks for embedding.
 *
 * Strategy: sentence-aware sliding window
 *   - chunk_size  : ~300 tokens (~1500 chars)
 *   - overlap     : ~50 tokens  (~250 chars)
 */

const CHUNK_SIZE    = 1500; // characters (~300 tokens for English)
const CHUNK_OVERLAP = 250;  // characters overlap between chunks

/**
 * Split text into overlapping chunks.
 * Tries to break at sentence boundaries (. ! ?) for cleaner chunks.
 *
 * @param {string} text
 * @returns {string[]}
 */
export function chunkText(text) {
  const cleaned = text.replace(/\s+/g, ' ').trim();
  if (cleaned.length <= CHUNK_SIZE) return [cleaned];

  const chunks = [];
  let start = 0;

  while (start < cleaned.length) {
    let end = start + CHUNK_SIZE;

    if (end >= cleaned.length) {
      chunks.push(cleaned.slice(start).trim());
      break;
    }

    // Try to find a sentence boundary near the end of the window
    const window = cleaned.slice(start, end + 100); // look a bit ahead
    const sentenceEnd = findSentenceBoundary(window, CHUNK_SIZE);

    if (sentenceEnd > CHUNK_SIZE * 0.5) {
      end = start + sentenceEnd;
    }

    const chunk = cleaned.slice(start, end).trim();
    if (chunk.length > 50) chunks.push(chunk); // skip tiny chunks

    // Move forward with overlap
    start = end - CHUNK_OVERLAP;
    if (start < 0) start = 0;
  }

  return chunks;
}

/**
 * Find the last sentence boundary (. ! ?) before or near `targetPos`.
 */
function findSentenceBoundary(text, targetPos) {
  // Search backwards from targetPos for . ! ?
  for (let i = Math.min(targetPos + 50, text.length - 1); i >= targetPos * 0.5; i--) {
    if ('.!?'.includes(text[i]) && (i + 1 >= text.length || text[i + 1] === ' ')) {
      return i + 1;
    }
  }
  // Fallback: break at last space
  for (let i = targetPos; i >= targetPos * 0.7; i--) {
    if (text[i] === ' ') return i;
  }
  return targetPos;
}

/**
 * Chunk a full document record into multiple chunk objects.
 *
 * @param {{ source, category, title, content, symptoms, precautions, metadata }} doc
 * @returns {Array<{ doc_id, source, category, title, chunk_index, content, metadata }>}
 */
export function chunkDocument(doc) {
  // Build the full text to chunk — include structured fields for richer context
  const parts = [
    `Title: ${doc.title}`,
    `Category: ${doc.category}`,
    doc.content || '',
  ];

  if (doc.symptoms && doc.symptoms.length > 0) {
    parts.push(`Symptoms: ${doc.symptoms.join(', ')}`);
  }
  if (doc.precautions && doc.precautions.length > 0) {
    parts.push(`Precautions: ${doc.precautions.join('. ')}`);
  }

  const fullText = parts.filter(Boolean).join('\n');
  const textChunks = chunkText(fullText);

  // Stable doc_id from source + title
  const doc_id = `${doc.source}::${doc.title}`.replace(/\s+/g, '_').slice(0, 200);

  return textChunks.map((content, chunk_index) => ({
    doc_id,
    source:      doc.source,
    category:    doc.category,
    title:       doc.title,
    chunk_index,
    content,
    metadata: {
      ...(doc.metadata || {}),
      total_chunks: textChunks.length,
    },
  }));
}
