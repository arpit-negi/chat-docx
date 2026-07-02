/**
 * UPLOAD API ROUTE
 * ================
 * In Next.js App Router, any file named "route.js" inside app/api/ becomes
 * a backend API endpoint automatically. No server setup needed.
 *
 * This file handles: POST /api/upload
 *
 * WHAT IS AN API ROUTE?
 * When the browser sends a request to "/api/upload", Next.js runs this file
 * on the SERVER (not in the browser). It has access to the file system,
 * can read secrets, and does the heavy lifting.
 *
 * The browser cannot run code like "read this file from disk" — that's a
 * security risk. That's why we split frontend (browser) and backend (server).
 */

import { NextResponse } from 'next/server';
import { v4 as uuidv4 } from 'uuid';
import { extractText, chunkText } from '../../../lib/documentProcessor.js';
import { storeDocument } from '../../../lib/vectorStore.js';

// Map MIME types to our internal file type strings
const ALLOWED_TYPES = {
  'application/pdf': 'pdf',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
  'text/plain': 'txt',
};

export async function POST(request) {
  try {
    // Parse the multipart form data (how browsers send files)
    const formData = await request.formData();
    const file = formData.get('file');

    if (!file) {
      return NextResponse.json({ error: 'No file provided' }, { status: 400 });
    }

    // Determine file type from MIME type or extension
    let fileType = ALLOWED_TYPES[file.type];
    if (!fileType) {
      const ext = file.name.split('.').pop().toLowerCase();
      const extMap = { pdf: 'pdf', docx: 'docx', txt: 'txt' };
      fileType = extMap[ext];
    }

    if (!fileType) {
      return NextResponse.json(
        { error: 'Unsupported file type. Please upload PDF, DOCX, or TXT.' },
        { status: 400 }
      );
    }

    // Convert the file to a Buffer (raw bytes) so our libraries can read it
    // file.arrayBuffer() = reads the file contents into memory as raw bytes
    const arrayBuffer = await file.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);

    // Step 1: Extract plain text from the file
    const text = await extractText(buffer, fileType);

    if (!text || text.trim().length < 50) {
      return NextResponse.json(
        { error: 'Could not extract meaningful text. The file may be empty or image-based (scanned PDF).' },
        { status: 400 }
      );
    }

    // Step 2: Split text into chunks
    // Smart chunk sizing based on document length:
    // Short docs (resumes, 1-5 pages) → bigger chunks preserve full context
    // Long docs (books, reports, 20+ pages) → smaller chunks = precise retrieval
    const docLength = text.length;
    let chunkSize, overlap;
    if (docLength < 5000) {
      chunkSize = 1000; overlap = 200; // short doc: keep more context per chunk
    } else if (docLength < 30000) {
      chunkSize = 800;  overlap = 150; // medium doc: balanced
    } else {
      chunkSize = 500;  overlap = 100; // long doc: precise retrieval matters more
    }
    const chunks = chunkText(text, chunkSize, overlap);

    if (chunks.length === 0) {
      return NextResponse.json({ error: 'Document produced no text chunks.' }, { status: 400 });
    }

    // Step 3: Generate a unique ID for this document
    const documentId = uuidv4();

    // Step 4: Embed chunks and save to disk
    // This is the slow step (1-30 seconds depending on document size)
    const chunkCount = await storeDocument(documentId, file.name, chunks);

    return NextResponse.json({
      documentId,
      filename: file.name,
      chunkCount,
      message: `Ready! Indexed ${chunkCount} chunks from your document.`,
    });

  } catch (error) {
    console.error('Upload error:', error);
    return NextResponse.json(
      { error: `Processing failed: ${error.message}` },
      { status: 500 }
    );
  }
}
