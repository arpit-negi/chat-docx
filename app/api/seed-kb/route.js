/**
 * SEED KNOWLEDGE BASE ROUTE
 * =========================
 * POST /api/seed-kb
 *
 * Lets you define your own Q&A pairs for a document.
 * These are answered INSTANTLY (no LLM, no tokens) forever.
 *
 * HOW TO CALL IT (from browser console or Postman):
 *
 * fetch('/api/seed-kb', {
 *   method: 'POST',
 *   headers: { 'Content-Type': 'application/json' },
 *   body: JSON.stringify({
 *     documentId: 'your-document-id-here',
 *     pairs: [
 *       { question: "What is your name?", answer: "Arpit Negi" },
 *       { question: "What is your degree?", answer: "B.Tech Computer Science" },
 *     ]
 *   })
 * })
 *
 * PYTHON EQUIVALENT:
 * import requests
 * requests.post('http://localhost:3000/api/seed-kb', json={
 *   'documentId': 'your-id',
 *   'pairs': [
 *     {'question': 'What is your name?', 'answer': 'Arpit Negi'},
 *   ]
 * })
 */

import { NextResponse } from 'next/server';
import { seedKnowledgeBase, getKnowledgeBaseStats } from '../../../lib/knowledgeBase.js';

export async function POST(request) {
  try {
    const { documentId, pairs } = await request.json();

    if (!documentId || !Array.isArray(pairs) || pairs.length === 0) {
      return NextResponse.json(
        { error: 'Provide documentId and pairs array: [{question, answer}]' },
        { status: 400 }
      );
    }

    // Validate pairs
    for (const pair of pairs) {
      if (!pair.question?.trim() || !pair.answer?.trim()) {
        return NextResponse.json(
          { error: 'Each pair must have a non-empty question and answer.' },
          { status: 400 }
        );
      }
    }

    await seedKnowledgeBase(documentId, pairs);
    const stats = await getKnowledgeBaseStats(documentId);

    return NextResponse.json({
      message: `Successfully added ${pairs.length} Q&A pairs to knowledge base.`,
      stats,
    });

  } catch (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}

// GET /api/seed-kb?documentId=xxx → returns stats
export async function GET(request) {
  try {
    const { searchParams } = new URL(request.url);
    const documentId = searchParams.get('documentId');

    if (!documentId) {
      return NextResponse.json({ error: 'Provide ?documentId=xxx' }, { status: 400 });
    }

    const stats = await getKnowledgeBaseStats(documentId);
    return NextResponse.json(stats);

  } catch (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
