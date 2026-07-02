/**
 * ASK API ROUTE — with Semantic Cache + Knowledge Base
 * =====================================================
 * NEW FLOW:
 *
 *   User question
 *         ↓
 *   [1] Check Knowledge Base (instant, free)
 *         ↓ match found (similarity > 0.85)
 *         → Return cached answer ✓ (0 tokens used)
 *
 *         ↓ no match
 *   [2] Full RAG pipeline (retrieval + LLM)
 *         ↓
 *         → Return LLM answer
 *         → Auto-cache this answer for next time ✓
 */

import { NextResponse } from 'next/server';
import { searchChunks } from '../../../lib/vectorStore.js';
import {
  searchKnowledgeBase,
  addToKnowledgeBase,
} from '../../../lib/knowledgeBase.js';

// Lazy initialization — only create client when a request comes in
// This prevents build-time failures when env vars aren't available
let groq = null;
async function getGroq() {
  if (!groq) {
    const { default: Groq } = await import('groq-sdk');
    groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
  }
  return groq;
}

export async function POST(request) {
  try {
    const { documentId, question } = await request.json();

    if (!documentId || !question?.trim()) {
      return NextResponse.json(
        { error: 'Both documentId and question are required.' },
        { status: 400 }
      );
    }

    // ─── STEP 1: CHECK KNOWLEDGE BASE FIRST ──────────────────────────────
    const cached = await searchKnowledgeBase(documentId, question);

    if (cached) {
      console.log(`Cache HIT (${(cached.similarity * 100).toFixed(0)}% match): "${cached.question}"`);
      return NextResponse.json({
        answer: cached.answer,
        sourceChunks: [],
        fromCache: true,
        cacheSource: cached.source,
        cacheSimilarity: cached.similarity,
        matchedQuestion: cached.question,
        tokensUsed: 0,
      });
    }

    console.log(`Cache MISS — calling Groq LLM for: "${question}"`);

    // ─── STEP 2: RETRIEVE relevant chunks ────────────────────────────────
    const relevantChunks = await searchChunks(documentId, question, 5);

    if (relevantChunks.length === 0) {
      return NextResponse.json({
        answer: "I couldn't find relevant information in the document. Try rephrasing your question.",
        sourceChunks: [],
        fromCache: false,
      });
    }

    // ─── STEP 3: AUGMENT — build prompt with context ─────────────────────
    // No [Excerpt N] labels — they cause the AI to say "as stated in Excerpt 1"
    const contextText = relevantChunks.join('\n\n---\n\n');

    // ─── STEP 4: GENERATE — call Groq LLM ────────────────────────────────
    const response = await (await getGroq()).chat.completions.create({
      model: 'llama-3.3-70b-versatile',
      messages: [
        {
          role: 'system',
          content: `You are a smart personal assistant. The user has uploaded a document (usually a resume or profile) and you have access to its content.

Behave like this:

1. DOCUMENT FACTS ("what is my name?", "what are my skills?", "where did I study?")
   → Answer strictly from the document content provided.

2. MIXED QUESTIONS ("based on my skills, what should I learn?", "am I ready for a senior role?")
   → Read the person's info from the document, then use your own knowledge to give real, helpful advice tailored to them.

3. GENERAL WORLD QUESTIONS ("what are companies asking in interviews?", "what is the job market like?", "what skills are trending?")
   → Answer from your own knowledge directly and helpfully. Do not say the document doesn't have this — the user knows that. They are just asking you a general question.

4. Never mention excerpts, document sections, or reference numbers. Talk like a human assistant, not a robot reading a file.

5. Never refuse to answer a general knowledge question just because it's not in the document. You are allowed to use your full knowledge.`,
        },
        {
          role: 'user',
          content: `Here is content from the user's document:\n\n${contextText}\n\n---\n\nUser question: ${question}`,
        },
      ],
      temperature: 0.1,
      max_tokens: 1024,
      top_p: 0.9,
    });

    const answer = response.choices[0].message.content;
    const tokensUsed = response.usage?.total_tokens || 0;

    // ─── STEP 5: AUTO-CACHE the answer ───────────────────────────────────
    addToKnowledgeBase(documentId, question, answer, 'llm_cached')
      .catch(err => console.warn('Cache write failed (non-fatal):', err.message));

    return NextResponse.json({
      answer,
      sourceChunks: relevantChunks,
      fromCache: false,
      tokensUsed,
    });

  } catch (error) {
    console.error('Ask error:', error);

    let message = error.message;
    if (message.includes('401') || message.includes('invalid_api_key')) {
      message = 'Invalid Groq API key. Check your GROQ_API_KEY in .env.local';
    } else if (message.includes('429') || message.includes('rate_limit')) {
      message = 'Rate limit hit. Wait a moment and try again.';
    }

    return NextResponse.json({ error: message }, { status: 500 });
  }
}
