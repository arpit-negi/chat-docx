/**
 * DOCUMENT PROCESSOR
 * ==================
 * One job: take a file's raw bytes and return clean text.
 *
 * WHY DO WE NEED TEXT EXTRACTION?
 * A PDF is NOT a text file. It's a format designed for PRINTING — it stores
 * character positions, fonts, and page layout as binary data. We can't just
 * read it like a .txt file. We need a library that understands the PDF format
 * and pulls out just the words.
 *
 * Same with DOCX — it's actually a ZIP file containing XML inside. mammoth
 * unzips it, parses the XML, and gives us plain text.
 *
 * WHY SPLIT INTO CHUNKS?
 * Claude (and all AI models) have a limit on how much text they can read at
 * once. This limit is called the "context window". Claude Haiku can handle
 * about 200,000 tokens (~150,000 words) — but we can't send the WHOLE document
 * for every question. That would be:
 *   1. Slow (more text = slower response)
 *   2. Expensive (you pay per token in most APIs)
 *   3. Less accurate (the model can get "lost" in too much irrelevant text)
 *
 * Instead, we find only the RELEVANT parts and send just those. That's what
 * chunking + vector search enables.
 */

/**
 * Extracts plain text from a file Buffer (raw bytes).
 * @param {Buffer} buffer - The file contents as raw bytes
 * @param {string} fileType - "pdf", "docx", or "txt"
 * @returns {Promise<string>} - The extracted plain text
 */
export async function extractText(buffer, fileType) {
  if (fileType === 'pdf') {
    // pdf-parse reads the binary PDF format and returns text
    // We import it dynamically because it's a server-only module
    const pdfParse = (await import('pdf-parse/lib/pdf-parse.js')).default;
    const result = await pdfParse(buffer);
    return result.text;
  }

  if (fileType === 'docx') {
    // mammoth converts DOCX → plain text (or HTML if you want formatting)
    const mammoth = (await import('mammoth')).default;
    const result = await mammoth.extractRawText({ buffer });
    return result.value;
  }

  if (fileType === 'txt') {
    // TXT files are already plain text — just decode the bytes
    // 'utf-8' is the most common text encoding (handles letters from all languages)
    return buffer.toString('utf-8');
  }

  throw new Error(`Unsupported file type: ${fileType}`);
}

/**
 * Splits a long text into smaller overlapping chunks.
 *
 * VISUAL EXAMPLE of overlap (chunkSize=20, overlap=5):
 *
 * Full text: "The quick brown fox jumps over the lazy dog"
 *
 * Chunk 1:   "The quick brown fox "
 * Chunk 2:               "fox jumps over the la"
 * Chunk 3:                           "the lazy dog"
 *
 * The words "fox" and "the" appear in two chunks each.
 * This prevents important context from being cut off at chunk boundaries.
 *
 * @param {string} text - The full document t-ext
 * @param {number} chunkSize - Characters per chunk (default: 500 ≈ 80 words)
 * @param {number} overlap - Characters to repeat between chunks (default: 100)
 * @returns {string[]} - Array of text chunks
 */
export function chunkText(text, chunkSize = 500, overlap = 100) {
  // Clean up the text: collapse 3+ newlines to 2, and remove double spaces
  const cleaned = text
    .replace(/\n{3,}/g, '\n\n')
    .replace(/ {2,}/g, ' ')
    .trim();

  if (!cleaned) return [];

  const chunks = [];
  let start = 0;

  while (start < cleaned.length) {
    let end = start + chunkSize;

    // Don't cut in the middle of a word — find the last space or period
    if (end < cleaned.length) {
      const breakPoint = cleaned.lastIndexOf('. ', end);
      const newlinePoint = cleaned.lastIndexOf('\n', end);
      const best = Math.max(breakPoint, newlinePoint);
      if (best > start + overlap) {
        end = best + 1;
      }
    }

    const chunk = cleaned.slice(start, end).trim();
    if (chunk.length > 20) { // skip tiny fragments
      chunks.push(chunk);
    }

    // Step forward by (chunkSize - overlap) so the next chunk overlaps
    start += chunkSize - overlap;
  }

  return chunks;
}
