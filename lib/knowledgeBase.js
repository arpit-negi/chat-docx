/**
 * KNOWLEDGE BASE + SEMANTIC CACHE
 * ================================
 * This module implements two things:
 *
 * 1. KNOWLEDGE BASE — a predefined set of Q&A pairs that we answer
 *    instantly without any LLM call. You define these yourself based on
 *    what questions you KNOW users will ask about your document.
 *
 * 2. SEMANTIC CACHE — every answer the LLM gives is stored here too.
 *    If another user asks the same (or very similar) question, we
 *    return the cached answer instead of calling the LLM again.
 *
 * HOW THE MATCHING WORKS:
 * We embed every "known question" once at startup. When a new query
 * comes in, we embed it and compare against all known question embeddings.
 * If the similarity is above a threshold (0.85), we consider it a match.
 *
 * WHY 0.85 AS THRESHOLD?
 *   0.95 = too strict (only catches near-identical wording)
 *   0.85 = catches paraphrases ("what's my name" vs "tell me my name")
 *   0.70 = too loose (might match unrelated questions)
 *
 * REAL WORLD ANALOGY:
 * Think of it like a customer service FAQ page. 80% of customers ask
 * the same 20 questions. You write answers for those 20 questions once.
 * Only truly unique questions get escalated to a human (the LLM).
 *
 * PYTHON EQUIVALENT:
 * In Python you'd use the same logic:
 *   from sentence_transformers import SentenceTransformer, util
 *   model = SentenceTransformer('all-MiniLM-L6-v2')
 *   known_embeddings = model.encode(known_questions)
 *   query_embedding = model.encode([user_question])
 *   scores = util.cos_sim(query_embedding, known_embeddings)
 *   best_idx = scores.argmax()
 *   if scores[0][best_idx] > 0.85:
 *       return known_answers[best_idx]
 */

import fs from 'fs/promises';
import path from 'path';
import { embedTexts, cosineSimilarity } from './embeddings.js';

const KB_DIR = path.join(process.cwd(), 'data');
const KB_FILE = path.join(KB_DIR, 'knowledge_base.json');

// In-memory cache of embeddings so we don't re-embed on every request
// (embeddings are computed once, then reused)
let embeddingCache = null;

/**
 * The structure of a knowledge base entry:
 * {
 *   question: "What is your name?",      ← the canonical question
 *   answer: "My name is Arpit Negi.",    ← the pre-written answer
 *   aliases: ["whats my name", "who am I", "tell me my name"],  ← variants
 *   embedding: [0.82, 0.15, ...]         ← computed and stored once
 * }
 */

async function ensureKBDir() {
  await fs.mkdir(KB_DIR, { recursive: true });
}

/**
 * Loads the knowledge base from disk.
 * Returns empty structure if file doesn't exist yet.
 */
export async function loadKnowledgeBase(documentId) {
  const kbPath = path.join(KB_DIR, `kb_${documentId}.json`);
  try {
    const raw = await fs.readFile(kbPath, 'utf-8');
    return JSON.parse(raw);
  } catch {
    return { documentId, entries: [] };
  }
}

/**
 * Saves the knowledge base to disk.
 */
async function saveKnowledgeBase(documentId, kb) {
  await ensureKBDir();
  const kbPath = path.join(KB_DIR, `kb_${documentId}.json`);
  await fs.writeFile(kbPath, JSON.stringify(kb, null, 2));
}

/**
 * Adds a new Q&A pair to the knowledge base.
 * Computes and stores the embedding for the question so future
 * lookups are instant (no re-embedding needed).
 *
 * @param {string} documentId
 * @param {string} question - The question to store
 * @param {string} answer   - The answer to store
 * @param {string} source   - 'predefined' | 'llm_cached'
 */
export async function addToKnowledgeBase(documentId, question, answer, source = 'llm_cached') {
  const kb = await loadKnowledgeBase(documentId);

  // Embed the question — stored so we never need to embed it again
  const [embedding] = await embedTexts([question]);

  kb.entries.push({
    question,
    answer,
    source,         // track whether this was predefined or cached from LLM
    embedding,
    addedAt: new Date().toISOString(),
    hitCount: 0,    // how many times this cache entry was used
  });

  // Reset in-memory cache so next lookup uses updated data
  embeddingCache = null;

  await saveKnowledgeBase(documentId, kb);
}

/**
 * Seeds the knowledge base with predefined Q&A pairs.
 * Call this after uploading a document if you know common questions.
 *
 * WHY SEED IT?
 * The first time a user asks "what is my name?", it will go to the LLM.
 * That answer gets cached. The 2nd user gets the cached answer.
 * But if you KNOW the answer already (it's in the document you uploaded),
 * you can pre-fill these to make ALL requests instant from day one.
 *
 * @param {string} documentId
 * @param {Array<{question: string, answer: string}>} pairs
 */
export async function seedKnowledgeBase(documentId, pairs) {
  for (const { question, answer } of pairs) {
    await addToKnowledgeBase(documentId, question, answer, 'predefined');
  }
}

/**
 * Searches the knowledge base for a matching question.
 *
 * @param {string} documentId
 * @param {string} userQuestion
 * @param {number} threshold - Minimum similarity to count as a match (0-1)
 * @returns {{ answer: string, similarity: number, source: string } | null}
 */
export async function searchKnowledgeBase(documentId, userQuestion, threshold = 0.85) {
  const kb = await loadKnowledgeBase(documentId);

  if (kb.entries.length === 0) return null;

  // Embed the user's question
  const [queryEmbedding] = await embedTexts([userQuestion]);

  // Compare against all stored question embeddings
  let bestMatch = null;
  let bestScore = -1;

  for (const entry of kb.entries) {
    if (!entry.embedding) continue;

    const score = cosineSimilarity(queryEmbedding, entry.embedding);

    if (score > bestScore) {
      bestScore = score;
      bestMatch = entry;
    }
  }

  // Only return if similarity is above threshold
  if (bestScore >= threshold && bestMatch) {
    // Update hit count (useful for analytics — see which answers are popular)
    bestMatch.hitCount = (bestMatch.hitCount || 0) + 1;
    const kbPath = path.join(KB_DIR, `kb_${documentId}.json`);
    await fs.writeFile(kbPath, JSON.stringify(kb, null, 2));

    return {
      answer: bestMatch.answer,
      similarity: bestScore,
      source: bestMatch.source,
      question: bestMatch.question, // the original question it matched
    };
  }

  return null; // no match — caller should use LLM
}

/**
 * Returns analytics: which questions are asked most, cache hit rate, etc.
 * Useful for understanding your users and optimizing the knowledge base.
 */
export async function getKnowledgeBaseStats(documentId) {
  const kb = await loadKnowledgeBase(documentId);
  const total = kb.entries.reduce((sum, e) => sum + (e.hitCount || 0), 0);

  return {
    totalEntries: kb.entries.length,
    predefined: kb.entries.filter(e => e.source === 'predefined').length,
    cached: kb.entries.filter(e => e.source === 'llm_cached').length,
    totalCacheHits: total,
    topQuestions: kb.entries
      .sort((a, b) => (b.hitCount || 0) - (a.hitCount || 0))
      .slice(0, 5)
      .map(e => ({ question: e.question, hits: e.hitCount || 0, source: e.source })),
  };
}
