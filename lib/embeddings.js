/**
 * EMBEDDINGS
 * ==========
 * This is the most important concept in RAG to understand.
 *
 * WHAT IS AN EMBEDDING?
 * An embedding is a way to represent TEXT as NUMBERS so that a computer
 * can measure how similar two pieces of text are.
 *
 * Think of it like this: imagine placing every possible sentence on a
 * giant map. Similar sentences are placed near each other. Very different
 * sentences are placed far apart.
 *
 *       [animals]
 *    "The cat sat"  ←→  "A dog ran"   (close together, same topic)
 *
 *                                      "Stock market rose"  (far away)
 *
 * An embedding is the X,Y coordinate of a sentence on that map.
 * But instead of 2 coordinates (X, Y), we use 384 coordinates.
 * We can't visualize 384 dimensions, but math works fine with them.
 *
 * EXAMPLE (simplified to 3 numbers):
 *   "The cat sat on the mat"   → [0.82, 0.15, 0.33]
 *   "A feline rested on a rug" → [0.80, 0.14, 0.35]  ← SIMILAR (same meaning)
 *   "Stock markets fell today"  → [0.02, 0.91, 0.12]  ← DIFFERENT
 *
 * HOW WE USE THIS FOR RAG:
 * 1. Embed every document chunk and store the numbers
 * 2. When user asks a question, embed the question too
 * 3. Find which chunk embeddings are CLOSEST to the question embedding
 * 4. Those closest chunks = most relevant to the question
 *
 * WHY @xenova/transformers?
 * This is HuggingFace's JavaScript library. It downloads a pre-trained AI
 * model (23MB, quantized version of all-MiniLM-L6-v2) that was trained to
 * create these embeddings. It runs 100% on YOUR machine — no API key, no cost.
 *
 * The model downloads to a cache folder the first time you run the app.
 * After that, it loads from cache (takes ~1 second).
 *
 * WHY A SINGLETON (single instance)?
 * Loading the model takes 1-3 seconds. If we loaded it fresh on every API
 * request, every question would be delayed by 3 seconds. Instead, we load
 * it ONCE when the server starts, and reuse the same instance forever.
 */

let pipeline = null;
let embedder = null;

/**
 * Loads the embedding model once and caches it.
 * All subsequent calls return the cached model instantly.
 */
async function getEmbedder() {
  if (embedder) return embedder;

  console.log('Loading embedding model (first run downloads ~23MB from HuggingFace)...');

  // Dynamic import because this is a server-only module
  const { pipeline: createPipeline } = await import('@huggingface/transformers');

  // 'feature-extraction' = we want the embedding vectors (not text generation)
  // 'Xenova/all-MiniLM-L6-v2' = the model name on HuggingFace Hub
  //   - all = trained on all kinds of text (not just one domain)
  //   - MiniLM = "Mini Language Model" (small and fast)
  //   - L6 = 6 transformer layers
  //   - v2 = version 2
  // QUALITY LADDER — uncomment the model you want (higher = better but slower):
  // Level 1 — fastest, 23MB:  'Xenova/all-MiniLM-L6-v2'      (384 dims)
  // Level 2 — balanced, 90MB: 'Xenova/all-mpnet-base-v2'      (768 dims)
  // Level 3 — best local:     'Xenova/multi-qa-mpnet-base-cos-v1' (768 dims, trained for Q&A)
  embedder = await createPipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

  console.log('Embedding model ready!');
  return embedder;
}

/**
 * Converts an array of text strings into an array of embedding vectors.
 *
 * @param {string[]} texts - Array of text strings to embed
 * @returns {Promise<number[][]>} - Array of embedding vectors (each is 384 numbers)
 */
export async function embedTexts(texts) {
  const model = await getEmbedder();
  const embeddings = [];

  // Process in batches of 32 to avoid memory issues on large documents
  const batchSize = 32;
  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);

    // The model processes each text and returns a tensor (multi-dimensional array)
    for (const text of batch) {
      const output = await model(text, { pooling: 'mean', normalize: true });
      // pooling: 'mean' = average all token embeddings into one vector
      // normalize: true = scale the vector to length 1 (needed for cosine similarity)
      embeddings.push(Array.from(output.data));
    }
  }

  return embeddings;
}

/**
 * Measures how similar two vectors are using cosine similarity.
 *
 * WHAT IS COSINE SIMILARITY?
 * It measures the ANGLE between two vectors. Think of two arrows pointing
 * in space — if they point in almost the same direction, the angle is small,
 * and the similarity is close to 1. If they point in opposite directions,
 * similarity is -1. Perpendicular = 0.
 *
 * For text: similar meaning = arrows point same direction = high similarity.
 *
 * WHY COSINE AND NOT EUCLIDEAN (straight-line) DISTANCE?
 * A long document's chunks have bigger numbers (more content). Euclidean
 * distance would say they're "far away" just because of their size. Cosine
 * only cares about DIRECTION (meaning), not magnitude (size). That's better
 * for text.
 *
 * @param {number[]} a - First embedding vector
 * @param {number[]} b - Second embedding vector
 * @returns {number} - Similarity score between -1 and 1 (higher = more similar)
 */
export function cosineSimilarity(a, b) {
  // Dot product: multiply each pair of numbers and sum them up
  const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);

  // Magnitudes: the "length" of each vector
  const magA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const magB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));

  // Cosine similarity = dot product / (magnitude A × magnitude B)
  return dot / (magA * magB);
}
