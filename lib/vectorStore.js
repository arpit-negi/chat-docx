/**
 * VECTOR STORE
 * ============
 * Stores document chunks + their embeddings, and lets us search them.
 *
 * WHY NOT USE A REAL VECTOR DATABASE (like ChromaDB, Pinecone)?
 * Real vector databases are better for production (10,000+ documents), but
 * they add installation complexity. For learning, storing data in a JSON file
 * on disk is PERFECT because:
 *   1. You can open the file and SEE your data with your own eyes
 *   2. No extra software to install
 *   3. Easy to understand — it's just a JSON object
 *
 * WHEN SHOULD YOU UPGRADE TO A REAL VECTOR DB?
 * When you have many documents (50+), or when search becomes slow (>500 chunks).
 * For this project, a JSON file is ideal.
 *
 * HOW THE DATA LOOKS (what's stored in each JSON file):
 * {
 *   "documentId": "abc-123",
 *   "filename": "contract.pdf",
 *   "chunks": [
 *     {
 *       "id": 0,
 *       "text": "This agreement is entered into...",
 *       "embedding": [0.12, 0.34, 0.01, ...]  // 384 numbers
 *     },
 *     ...
 *   ]
 * }
 */

import fs from 'fs/promises';
import path from 'path';
import { embedTexts, cosineSimilarity } from './embeddings.js';

// Where we store document data on disk
// process.cwd() = the root folder of your Next.js project
const DATA_DIR = path.join(process.cwd(), 'data');

/** Makes sure the data directory exists before we try to write to it */
async function ensureDataDir() {
  await fs.mkdir(DATA_DIR, { recursive: true });
}

/** Returns the file path for a document's data file */
function getDocPath(documentId) {
  return path.join(DATA_DIR, `${documentId}.json`);
}

/**
 * Saves document chunks + their embeddings to disk.
 *
 * @param {string} documentId - Unique ID for this document
 * @param {string} filename - Original filename (for display)
 * @param {string[]} chunks - Array of text chunks
 * @returns {Promise<number>} - Number of chunks stored
 */
export async function storeDocument(documentId, filename, chunks) {
  await ensureDataDir();

  console.log(`Embedding ${chunks.length} chunks...`);

  // Convert all chunks to embedding vectors
  // This is the most time-consuming step (1-5 seconds depending on doc size)
  const embeddings = await embedTexts(chunks);

  // Build the data structure to save
  const data = {
    documentId,
    filename,
    createdAt: new Date().toISOString(),
    chunks: chunks.map((text, i) => ({
      id: i,
      text,
      embedding: embeddings[i],
    })),
  };

  // Save to disk as a formatted JSON file
  await fs.writeFile(getDocPath(documentId), JSON.stringify(data, null, 2));

  console.log(`Stored ${chunks.length} chunks for document ${documentId}`);
  return chunks.length;
}

/**
 * Searches for chunks most relevant to a question.
 *
 * HOW THE SEARCH WORKS:
 * 1. Convert the question to an embedding vector
 * 2. Compare that vector to every stored chunk's embedding
 * 3. Sort by similarity score (highest = most relevant)
 * 4. Return the top K results
 *
 * This is called "nearest neighbor search" — finding the stored vectors
 * nearest (most similar) to the query vector.
 *
 * @param {string} documentId - Which document to search in
 * @param {string} question - The user's question
 * @param {number} topK - How many chunks to return (default: 5)
 * @returns {Promise<string[]>} - Array of relevant text chunks
 */
export async function searchChunks(documentId, question, topK = 5) {
  // Dynamic topK: questions asking for lists/all/every need more chunks
  const needsMoreContext = /\b(all|every|list|summarize|overview|compare|total)\b/i.test(question);
  if (needsMoreContext) topK = Math.min(topK * 2, 10); // double it, cap at 10
  const docPath = getDocPath(documentId);

  // Check if document exists
  try {
    await fs.access(docPath);
  } catch {
    throw new Error(`Document ${documentId} not found. Please upload it again.`);
  }

  const raw = await fs.readFile(docPath, 'utf-8');
  const data = JSON.parse(raw);

  // Embed the question using the SAME model we used for chunks
  // Critical: both must be in the same "vector space" to be comparable
  const [questionEmbedding] = await embedTexts([question]);

  // Score every chunk against the question
  const scored = data.chunks.map((chunk) => ({
    text: chunk.text,
    score: cosineSimilarity(questionEmbedding, chunk.embedding),
  }));

  // Sort by similarity score, highest first
  scored.sort((a, b) => b.score - a.score);

  // Return the top K chunks. We do NOT filter by score threshold because
  // short queries like "my name" have low absolute scores even when the right
  // chunk exists — the threshold was incorrectly discarding valid results.
  // We always return the best matches we have.
  return scored
    .slice(0, topK)
    .map((c) => c.text);
}

/**
 * Returns metadata about a stored document (without loading all embeddings).
 */
export async function getDocumentInfo(documentId) {
  try {
    const raw = await fs.readFile(getDocPath(documentId), 'utf-8');
    const data = JSON.parse(raw);
    return {
      documentId: data.documentId,
      filename: data.filename,
      chunkCount: data.chunks.length,
      createdAt: data.createdAt,
    };
  } catch {
    return null;
  }
}
