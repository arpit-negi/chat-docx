/**
 * WHY THIS CONFIG FILE EXISTS:
 *
 * Next.js tries to bundle ALL your code for the browser by default.
 * But some packages CANNOT run in a browser — they need Node.js (the server):
 *
 * - pdf-parse: reads binary PDF files from disk → needs Node.js file system
 * - @xenova/transformers: loads a 23MB AI model from disk → needs Node.js
 *
 * "serverExternalPackages" tells Next.js: "don't try to bundle these for
 * the browser, keep them server-side only."
 *
 * Without this, you'd get cryptic errors like "fs is not defined"
 * (because 'fs' = file system = Node.js only, not available in browsers).
 */

/** @type {import('next').NextConfig} */
const nextConfig = {
  serverExternalPackages: ['@huggingface/transformers', 'pdf-parse', 'mammoth'],

  // Increase the server response timeout for large document uploads
  // Default is 30s, which might not be enough for embedding 100+ chunks
  experimental: {
    serverActions: {
      bodySizeLimit: '10mb',
    },
  },
};

export default nextConfig;
