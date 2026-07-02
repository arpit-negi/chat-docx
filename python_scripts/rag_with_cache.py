"""
RAG WITH SEMANTIC CACHE — Python Implementation
================================================
This script does EXACTLY what our Next.js app does but in pure Python.
Useful for:
  - Running as a batch job / script
  - Preprocessing documents offline
  - Understanding the logic without a web server
  - Data science / Jupyter notebooks

INSTALL REQUIREMENTS:
  pip install sentence-transformers groq pypdf python-docx numpy

USAGE:
  python rag_with_cache.py
"""

import json
import os
import uuid
from pathlib import Path
import numpy as np

# ─── LIBRARIES ──────────────────────────────────────────────────────────────
# sentence-transformers: same embedding model as our JS app
from sentence_transformers import SentenceTransformer

# groq: Python SDK for Groq API (same API as JS version)
from groq import Groq

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your_groq_api_key_here")
EMBED_MODEL  = "all-MiniLM-L6-v2"   # same model as JS app → 384 dimensions
LLM_MODEL    = "llama-3.3-70b-versatile"
CHUNK_SIZE   = 800                   # characters per chunk
OVERLAP      = 150                   # overlap between chunks
TOP_K        = 5                     # how many chunks to retrieve
KB_THRESHOLD = 0.85                  # similarity needed for cache hit
DATA_DIR     = Path("data")

# ─── LOAD MODELS (once, reused for all queries) ──────────────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)
groq_client = Groq(api_key=GROQ_API_KEY)
print("Models ready.")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: DOCUMENT PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text(file_path: str) -> str:
    """Extract plain text from PDF, DOCX, or TXT files."""
    path = Path(file_path)

    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        return "\n\n".join(
            page.extract_text() for page in reader.pages
            if page.extract_text()
        )

    elif path.suffix.lower() == ".docx":
        from docx import Document
        doc = Document(file_path)
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

    elif path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks.
    Same algorithm as lib/documentProcessor.js in the JS app.
    """
    import re
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            # Try to break at a sentence boundary
            break_at = text.rfind(". ", start + overlap, end)
            newline_at = text.rfind("\n", start + overlap, end)
            best = max(break_at, newline_at)
            if best > start + overlap:
                end = best + 1

        chunk = text[start:end].strip()
        if len(chunk) > 20:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: EMBEDDINGS & VECTOR STORE
# ═══════════════════════════════════════════════════════════════════════════════

def cosine_similarity(a: list, b: list) -> float:
    """
    Measures similarity between two vectors.
    Returns 1.0 (identical) to -1.0 (opposite).
    Same formula as lib/embeddings.js cosineSimilarity().
    """
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def index_document(file_path: str) -> str:
    """
    Full pipeline: extract → chunk → embed → save to disk.
    Returns a document_id you use for all future queries.
    """
    document_id = str(uuid.uuid4())
    DATA_DIR.mkdir(exist_ok=True)

    print(f"Extracting text from {file_path}...")
    text = extract_text(file_path)
    print(f"Extracted {len(text)} characters.")

    # Smart chunk sizing (same logic as JS app)
    if len(text) < 5000:
        chunk_size, overlap = 1000, 200
    elif len(text) < 30000:
        chunk_size, overlap = 800, 150
    else:
        chunk_size, overlap = 500, 100

    chunks = chunk_text(text, chunk_size, overlap)
    print(f"Created {len(chunks)} chunks.")

    print(f"Embedding {len(chunks)} chunks (this takes a moment)...")
    # sentence-transformers can embed all chunks in one batch — much faster
    # than embedding one-by-one
    embeddings = embedder.encode(chunks, show_progress_bar=True).tolist()

    # Save to disk as JSON (same format as JS app)
    doc_data = {
        "documentId": document_id,
        "filename": Path(file_path).name,
        "chunks": [
            {"id": i, "text": chunk, "embedding": emb}
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]
    }

    output_path = DATA_DIR / f"{document_id}.json"
    output_path.write_text(json.dumps(doc_data, indent=2))
    print(f"Saved to {output_path}")

    return document_id


def search_chunks(document_id: str, question: str, top_k=TOP_K) -> list[str]:
    """Find the most relevant chunks for a question."""
    doc_path = DATA_DIR / f"{document_id}.json"
    data = json.loads(doc_path.read_text())

    # Embed the question
    query_embedding = embedder.encode([question])[0].tolist()

    # Score every chunk
    scored = [
        (chunk["text"], cosine_similarity(query_embedding, chunk["embedding"]))
        for chunk in data["chunks"]
    ]

    # Sort by score descending, return top K
    scored.sort(key=lambda x: x[1], reverse=True)
    return [text for text, _ in scored[:top_k]]


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: KNOWLEDGE BASE (SEMANTIC CACHE)
# ═══════════════════════════════════════════════════════════════════════════════

def load_kb(document_id: str) -> dict:
    kb_path = DATA_DIR / f"kb_{document_id}.json"
    if kb_path.exists():
        return json.loads(kb_path.read_text())
    return {"documentId": document_id, "entries": []}


def save_kb(document_id: str, kb: dict):
    DATA_DIR.mkdir(exist_ok=True)
    kb_path = DATA_DIR / f"kb_{document_id}.json"
    kb_path.write_text(json.dumps(kb, indent=2))


def add_to_kb(document_id: str, question: str, answer: str, source="llm_cached"):
    """Cache a Q&A pair with its embedding."""
    kb = load_kb(document_id)
    embedding = embedder.encode([question])[0].tolist()
    kb["entries"].append({
        "question": question,
        "answer": answer,
        "source": source,
        "embedding": embedding,
        "hitCount": 0,
    })
    save_kb(document_id, kb)


def seed_kb(document_id: str, pairs: list[dict]):
    """
    Pre-populate the knowledge base with known Q&A pairs.
    These are answered instantly (0 tokens) forever.

    pairs = [
        {"question": "What is your name?", "answer": "Arpit Negi"},
        {"question": "What is your degree?", "answer": "B.Tech CS"},
    ]
    """
    print(f"Seeding {len(pairs)} Q&A pairs into knowledge base...")
    for pair in pairs:
        add_to_kb(document_id, pair["question"], pair["answer"], source="predefined")
    print("Knowledge base seeded.")


def search_kb(document_id: str, question: str, threshold=KB_THRESHOLD):
    """
    Check if the question matches a known Q&A pair.
    Returns the answer if match found, None if not.
    """
    kb = load_kb(document_id)
    if not kb["entries"]:
        return None

    query_emb = embedder.encode([question])[0].tolist()

    best_score = -1
    best_entry = None

    for entry in kb["entries"]:
        score = cosine_similarity(query_emb, entry["embedding"])
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_score >= threshold and best_entry:
        best_entry["hitCount"] = best_entry.get("hitCount", 0) + 1
        save_kb(document_id, kb)
        return {
            "answer": best_entry["answer"],
            "similarity": best_score,
            "source": best_entry["source"],
            "matched_question": best_entry["question"],
        }

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: RAG PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def ask(document_id: str, question: str) -> dict:
    """
    Full RAG pipeline with semantic cache.
    Same logic as app/api/ask/route.js.
    """
    # 1. Check knowledge base first (free, instant)
    cached = search_kb(document_id, question)
    if cached:
        print(f"  ✓ Cache HIT ({cached['similarity']*100:.0f}% match) — 0 tokens used")
        return {
            "answer": cached["answer"],
            "from_cache": True,
            "source": cached["source"],
            "tokens_used": 0,
        }

    print(f"  ✗ Cache MISS — calling Groq LLM...")

    # 2. Retrieve relevant chunks
    chunks = search_chunks(document_id, question)
    if not chunks:
        return {"answer": "This information is not in the document.", "from_cache": False}

    # 3. Build prompt
    context = "\n\n---\n\n".join(f"[Excerpt {i+1}]\n{c}" for i, c in enumerate(chunks))
    prompt = f"Document excerpts:\n\n{context}\n\n---\n\nQuestion: {question}\n\nAnswer based only on the excerpts above:"

    # 4. Call Groq
    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": """You are a precise document assistant. Answer questions strictly based on the provided document excerpts. Do not use any outside knowledge. If the answer is not in the excerpts, say: "This information is not in the document." Be concise."""
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content
    tokens = response.usage.total_tokens

    # 5. Auto-cache the answer
    add_to_kb(document_id, question, answer, source="llm_cached")
    print(f"  → Used {tokens} tokens. Answer cached for next time.")

    return {"answer": answer, "from_cache": False, "tokens_used": tokens}


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO — run this script directly to see it work
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # ── If you pass a file path, index it ──
    # Usage: python rag_with_cache.py resume.pdf
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        doc_id = index_document(file_path)
        print(f"\nDocument ID: {doc_id}")
        print("Save this ID — you need it for queries.\n")

        # Optionally seed the knowledge base with known Q&A
        # Edit these pairs to match YOUR document
        known_pairs = [
            {"question": "What is your name?",    "answer": "Arpit Negi"},
            {"question": "What is your degree?",  "answer": "B.Tech in Computer Science"},
        ]
        seed_kb(doc_id, known_pairs)

    else:
        # ── Demo mode: uses a hardcoded doc ID ──
        # Replace this with your actual document ID from the web app
        doc_id = input("Enter document ID: ").strip()
        if not doc_id:
            print("No document ID provided. Index a document first.")
            sys.exit(1)

    # Interactive Q&A loop
    print("\n=== RAG Chat (type 'quit' to exit) ===\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        result = ask(doc_id, question)
        source = "⚡ CACHE" if result["from_cache"] else "🤖 LLM"
        print(f"\n{source}: {result['answer']}\n")
