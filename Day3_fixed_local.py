# -*- coding: utf-8 -*-
"""
Day3_fixed.py — Corrected Day 3: Knowledge Base + Hybrid RAG (LOCAL VERSION)

CHANGES FROM ORIGINAL COLAB SCRIPT:
  1. PDF extraction switched from PyPDF2 -> PyMuPDF (fitz).
     PyPDF2 scrambled Arabic (RTL) text order in the Saudi policy PDFs
     (National Water Strategy, Water Resources Code). PyMuPDF preserves
     correct reading order for Arabic/RTL content.
  2. BM25 tokenization now normalizes Arabic text (strips diacritics,
     unifies alef/ta-marbuta variants) before splitting, so Arabic
     keyword matching isn't silently broken by character-form mismatches.
  3. Source list is split into POLICY_SOURCES / TECHNICAL_SOURCES /
     GOVERNANCE_SOURCES, and a scoped policy_search() + technical_search()
     + build_evidence_context() are included so Day 4's recommendation
     engine can reliably surface regulatory context instead of leaving it
     to chance in the shared top-k pool.
  4. No Colab dependency — reads PDFs from a local folder instead of the
     upload widget. Run with: python Day3_fixed.py --pdf_dir ./pdfs

SETUP (run once):
    pip install PyMuPDF rank_bm25 numpy sentence-transformers chromadb pyarabic

USAGE:
    python Day3_fixed.py --pdf_dir /path/to/your/pdf/folder
    (defaults to ./pdfs in the current directory if --pdf_dir is omitted)
"""

# ============================================================
# DAY 3 — KNOWLEDGE BASE + HYBRID RAG (FIXED, LOCAL)
# ============================================================

import os
import re
import sys
import glob
import json
import argparse
import numpy as np
import fitz  # PyMuPDF
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from datetime import datetime

try:
    from pyarabic.araby import strip_tashkeel, normalize_hamza, normalize_alef
    HAVE_PYARABIC = True
except ImportError:
    HAVE_PYARABIC = False

print("=" * 70)
print("📚 DAY 3 — KNOWLEDGE BASE + HYBRID SEARCH (FIXED, LOCAL)")
print("=" * 70)

# ============================================================
# 1. LOAD PDF FILES FROM A LOCAL FOLDER
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--pdf_dir", default="./pdfs",
                     help="Folder containing your 10 knowledge base PDFs")
args, _ = parser.parse_known_args()

PDF_DIR = args.pdf_dir

print(f"\n📁 Looking for PDFs in: {os.path.abspath(PDF_DIR)}")
print("   Expected files:")
print("   1. Khait_Plant_Sounds_2023.pdf")
print("   2. FAO_Crop_Water_Response_66.pdf")
print("   3. FAO_Irrigation_Manual.pdf")
print("   4. MEWA_Tomato_Irrigation_Guide.pdf")
print("   5. AIAdoptionFramework.pdf")
print("   6. ai-principles.pdf")
print("   7. Personal_Data_Protection_Law.pdf")
print("   8. Saudi_GAP.pdf")
print("   9. SaudiWaterResourcesCode.pdf")
print("   10. الاستراتيجية_الوطنية_للمياه_2030.pdf")

if not os.path.isdir(PDF_DIR):
    raise FileNotFoundError(
        f"❌ Folder not found: {PDF_DIR}\n"
        f"   Create it and place your 10 PDFs inside, or pass --pdf_dir /your/path"
    )

pdf_paths = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
pdf_files = [os.path.basename(p) for p in pdf_paths]

print(f"\n✅ Number of PDF files found: {len(pdf_files)}")

if len(pdf_files) < 10:
    print(f"⚠️ Warning: Expected 10 files, but only {len(pdf_files)} were found.")

for i, f in enumerate(pdf_files, 1):
    print(f"   {i}. {f}")

if len(pdf_files) == 0:
    raise ValueError(f"❌ No PDF files found in {PDF_DIR}.")

# Map filename -> full path, since we're no longer in Colab's flat upload dir
PDF_PATH_MAP = {os.path.basename(p): p for p in pdf_paths}

# ============================================================
# 2. EXTRACT TEXT PAGE BY PAGE  (FIXED: PyMuPDF, not PyPDF2)
# ============================================================
# PyMuPDF's "text" extraction mode respects visual/reading order far
# more reliably than PyPDF2 for RTL scripts. This is the actual fix
# for the garbled Arabic seen in Day 3's original output.

def extract_pdf_pages(pdf_filename):
    pages = []
    pdf_path = PDF_PATH_MAP[pdf_filename]
    try:
        doc = fitz.open(pdf_path)
        print(f"\n📄 {pdf_filename} → {len(doc)} pages")
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text and text.strip():
                pages.append({
                    "source": pdf_filename,
                    "page": page_number,
                    "text": text.strip()
                })
        doc.close()
    except Exception as e:
        print(f"⚠️ Error reading {pdf_path}: {e}")
    return pages

all_pages = []
print("\n" + "=" * 70)
print("📖 EXTRACTING TEXT")
print("=" * 70)

for pdf in pdf_files:
    pages = extract_pdf_pages(pdf)
    all_pages.extend(pages)

print(f"\n✅ Total pages with extracted text: {len(all_pages)}")

# Quick sanity check: flag pages that still look suspicious
# (e.g. mostly digits/symbols, which was a symptom of the old garbling)
def looks_suspicious(text, sample_len=300):
    sample = text[:sample_len]
    if not sample.strip():
        return True
    alnum_ratio = sum(c.isalpha() for c in sample) / max(len(sample), 1)
    return alnum_ratio < 0.3

suspicious = [p for p in all_pages if looks_suspicious(p["text"])]
if suspicious:
    print(f"\n⚠️ {len(suspicious)} pages still look low-quality after extraction.")
    print("   Review these manually — may need OCR (scanned pages).")
    for p in suspicious[:5]:
        print(f"   - {p['source']} p.{p['page']}")

# ============================================================
# 3. CREATE CHUNKS  (unchanged logic)
# ============================================================

def create_chunks(pages, chunk_size=700, overlap=100):
    chunks = []
    chunk_id = 0
    for page_data in pages:
        text = page_data["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()
            if len(chunk_text) > 100:
                chunks.append({
                    "id": f"chunk_{chunk_id}",
                    "text": chunk_text,
                    "source": page_data["source"],
                    "page": page_data["page"]
                })
                chunk_id += 1
            start += chunk_size - overlap
    return chunks

print("\n" + "=" * 70)
print("✂️ CREATING CHUNKS")
print("=" * 70)

chunks = create_chunks(all_pages, chunk_size=700, overlap=100)
print(f"✅ Number of chunks: {len(chunks)}")

# ============================================================
# 4. BM25 KEYWORD SEARCH  (FIXED: Arabic-aware tokenization)
# ============================================================

def normalize_for_bm25(text):
    """Lowercase for Latin script; normalize Arabic forms so equivalent
    words don't get treated as different tokens."""
    text = text.lower()
    if HAVE_PYARABIC:
        text = strip_tashkeel(text)
        text = normalize_hamza(text)
        text = normalize_alef(text)
    else:
        # Fallback manual normalization if pyarabic isn't available
        text = re.sub(r"[\u064B-\u065F\u0670]", "", text)  # strip diacritics
        text = re.sub(r"[إأآا]", "ا", text)                 # unify alef forms
        text = re.sub(r"ة", "ه", text)                       # ta marbuta -> ha
    return text

print("\n" + "=" * 70)
print("🔎 BUILDING BM25 KEYWORD SEARCH")
print("=" * 70)

chunk_texts = [c["text"] for c in chunks]
tokenized_chunks = [normalize_for_bm25(t).split() for t in chunk_texts]
bm25 = BM25Okapi(tokenized_chunks)
print("✅ BM25 ready.")
if not HAVE_PYARABIC:
    print("ℹ️ pyarabic not installed — used regex fallback normalization.")

# ============================================================
# 5. LOAD EMBEDDING MODEL  (unchanged)
# ============================================================

print("\n" + "=" * 70)
print("🧠 LOADING EMBEDDING MODEL")
print("=" * 70)

embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
print("✅ Embedding model loaded.")

# ============================================================
# 6. CREATE EMBEDDINGS  (unchanged)
# ============================================================

print("\n" + "=" * 70)
print("🔢 CREATING EMBEDDINGS")
print("=" * 70)

embeddings = embedding_model.encode(
    chunk_texts,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)
print(f"✅ Embeddings shape: {embeddings.shape}")

# ============================================================
# 7. CREATE CHROMADB VECTOR DATABASE  (unchanged)
# ============================================================

print("\n" + "=" * 70)
print("🗄️ CREATING VECTOR DATABASE")
print("=" * 70)

client = chromadb.PersistentClient(path="./chroma_knowledge_base")
try:
    client.delete_collection("agri_knowledge_base")
except:
    pass

collection = client.create_collection("agri_knowledge_base")

collection.add(
    ids=[c["id"] for c in chunks],
    documents=chunk_texts,
    embeddings=[e.tolist() for e in embeddings],
    metadatas=[{"source": c["source"], "page": str(c["page"])} for c in chunks]
)

print(f"✅ ChromaDB ready. Chunks stored: {collection.count()}")

# ============================================================
# 8. SEARCH FUNCTIONS  (bm25/vector unchanged; hybrid unchanged;
#    NEW: policy_search / technical_search / build_evidence_context)
# ============================================================

def bm25_search(query, top_k=5):
    scores = bm25.get_scores(normalize_for_bm25(query).split())
    top = np.argsort(scores)[::-1][:top_k]
    results = []
    for r, idx in enumerate(top, 1):
        if scores[idx] > 0:
            results.append({
                "rank": r,
                "text": chunks[idx]["text"],
                "source": chunks[idx]["source"],
                "page": chunks[idx]["page"],
                "score": float(scores[idx])
            })
    return results

def vector_search(query, top_k=5):
    q_emb = embedding_model.encode([query], normalize_embeddings=True)[0]
    res = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    results = []
    for i in range(len(res["documents"][0])):
        results.append({
            "rank": i + 1,
            "text": res["documents"][0][i],
            "source": res["metadatas"][0][i]["source"],
            "page": res["metadatas"][0][i]["page"],
            "distance": res["distances"][0][i]
        })
    return results

def hybrid_search(query, top_k=5, candidate_k=10):
    bm25_res = bm25_search(query, top_k=candidate_k)
    vec_res = vector_search(query, top_k=candidate_k)

    rrf_scores = {}
    result_data = {}
    k = 60

    for r in bm25_res:
        cid = f"{r['source']}_{r['page']}"
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + r["rank"])
        result_data[cid] = {"text": r["text"], "source": r["source"], "page": r["page"]}

    for r in vec_res:
        cid = f"{r['source']}_{r['page']}"
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + r["rank"])
        if cid not in result_data:
            result_data[cid] = {"text": r["text"], "source": r["source"], "page": r["page"]}

    ranked = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
    results = []
    for rank, cid in enumerate(ranked, 1):
        results.append({
            "rank": rank,
            "hybrid_score": rrf_scores[cid],
            "text": result_data[cid]["text"],
            "source": result_data[cid]["source"],
            "page": result_data[cid]["page"]
        })
    return results

# --- NEW: dual-track retrieval for policy visibility (see Day 4 discussion) ---
# NOTE: match these strings EXACTLY to what appears in your uploaded filenames
# (check pdf_files above — some have a leading invisible LRM character).

POLICY_SOURCES = [
    s for s in pdf_files if any(k in s for k in
        ["الاستراتيجية", "قاب", "SaudiWaterResourcesCode"])
]
TECHNICAL_SOURCES = [
    s for s in pdf_files if any(k in s for k in
        ["FAO_Crop_Water_Response", "FAO_Irrigation_Manual",
         "MEWA_Tomato_Irrigation_Guide", "Khait_Plant_Sounds"])
]
GOVERNANCE_SOURCES = [
    s for s in pdf_files if any(k in s for k in
        ["ai-principles", "AIAdoptionFramework", "Personal Data"])
]

POLICY_RELEVANCE_THRESHOLD = 0.70  # tune against real queries before demo day

def policy_search(query, top_k=2, distance_threshold=POLICY_RELEVANCE_THRESHOLD):
    if not POLICY_SOURCES:
        return []
    q_emb = embedding_model.encode([query], normalize_embeddings=True)[0]
    res = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=top_k,
        where={"source": {"$in": POLICY_SOURCES}},
        include=["documents", "metadatas", "distances"],
    )
    if not res["documents"][0]:
        return []
    results = []
    for i in range(len(res["documents"][0])):
        distance = res["distances"][0][i]
        if distance <= distance_threshold:
            results.append({
                "rank": i + 1,
                "text": res["documents"][0][i],
                "source": res["metadatas"][0][i]["source"],
                "page": res["metadatas"][0][i]["page"],
                "distance": distance,
            })
    return results

def technical_search(query, top_k=3):
    if not TECHNICAL_SOURCES:
        return []
    q_emb = embedding_model.encode([query], normalize_embeddings=True)[0]
    res = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=top_k,
        where={"source": {"$in": TECHNICAL_SOURCES}},
        include=["documents", "metadatas", "distances"],
    )
    results = []
    for i in range(len(res["documents"][0])):
        results.append({
            "rank": i + 1,
            "text": res["documents"][0][i],
            "source": res["metadatas"][0][i]["source"],
            "page": res["metadatas"][0][i]["page"],
            "distance": res["distances"][0][i],
        })
    return results

# NOTE: GOVERNANCE_SOURCES (ai-principles.pdf, AIAdoptionFramework.pdf,
# Personal Data Protection Law) are intentionally NOT queried by any
# retrieval function. They stay indexed in ChromaDB (harmless, already
# embedded) but are treated as reference material to read directly when
# writing the Day 5/6 AI Readiness mapping and data-handling report
# sections — faster than building/tuning a retrieval function for a
# one-time write-up, and keeps them fully isolated from the live
# farmer-facing recommendation flow below.

def build_evidence_context(query):
    technical = technical_search(query, top_k=3)
    policy = policy_search(query, top_k=2)

    context = "TECHNICAL EVIDENCE:\n"
    for r in technical:
        context += f"- [{r['source']}, p.{r['page']}] {r['text'][:400]}...\n"

    if policy:
        context += "\nREGULATORY CONTEXT:\n"
        for r in policy:
            context += f"- [{r['source']}, p.{r['page']}] {r['text'][:400]}...\n"
    else:
        context += "\nREGULATORY CONTEXT: No directly relevant regulatory guidance found in the knowledge base for this query.\n"

    return context, technical, policy

# ============================================================
# 9. RUN AND SAVE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("🧪 RUNNING HYBRID SEARCH TESTS")
print("=" * 70)

test_queries = [
    "What irrigation practices are recommended for tomato plants?",
    "What are the signs of drought stress in plants?",
    "How can water resources be managed efficiently in agriculture?",
    "What agricultural practices support sustainable water use?",
    "ما هي ممارسات الري الموصى بها لنباتات الطماطم؟",
    "ما هي علامات الإجهاد المائي في النباتات؟"
]

all_results = []

for query in test_queries:
    print(f"\n🔍 Query: {query}")
    results = hybrid_search(query, top_k=3, candidate_k=10)
    all_results.append({"query": query, "results": results})
    for r in results:
        print(f"   Rank {r['rank']} → {r['source'][:40]}... (score: {r['hybrid_score']:.3f})")

# ============================================================
# 10. SAVE OUTPUTS  (unchanged, now includes policy/technical split in metadata)
# ============================================================

print("\n" + "=" * 70)
print("💾 SAVING OUTPUTS")
print("=" * 70)

metadata = {
    "timestamp": datetime.now().isoformat(),
    "pdf_files": pdf_files,
    "total_pdfs": len(pdf_files),
    "total_pages": len(all_pages),
    "total_chunks": len(chunks),
    "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
    "vector_db": "ChromaDB",
    "keyword_search": "BM25",
    "hybrid_method": "RRF",
    "extraction_method": "PyMuPDF (fitz)",
    "policy_sources": POLICY_SOURCES,
    "technical_sources": TECHNICAL_SOURCES,
    "governance_sources": GOVERNANCE_SOURCES,
    "suspicious_pages_flagged": len(suspicious),
}

with open("knowledge_base_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=4)

print("✅ metadata.json saved")

with open("Day3_Hybrid_Search_Results.txt", "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("DAY 3 — HYBRID SEARCH RESULTS (FIXED EXTRACTION)\n")
    f.write(f"Generated: {datetime.now().isoformat()}\n")
    f.write("=" * 70 + "\n\n")

    for entry in all_results:
        f.write(f"🔍 QUERY: {entry['query']}\n")
        f.write("-" * 50 + "\n")
        for r in entry["results"]:
            f.write(f"🏆 Rank {r['rank']}\n")
            f.write(f"📄 Source: {r['source']}\n")
            f.write(f"📑 Page: {r['page']}\n")
            f.write(f"⭐ Score: {r['hybrid_score']:.4f}\n")
            f.write(f"📋 Text: {r['text'][:500]}...\n\n")
        f.write("=" * 70 + "\n\n")

print("✅ Day3_Hybrid_Search_Results.txt saved")

# ============================================================
# 11. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("🎉 DAY 3 (FIXED) COMPLETED SUCCESSFULLY")
print("=" * 70)
print(f"""
✅ {len(pdf_files)} PDF files processed (PyMuPDF extraction)
✅ Page-level extraction — {len(suspicious)} pages flagged as low-quality
✅ Chunking (700 chars, 100 overlap)
✅ BM25 keyword search (Arabic-normalized)
✅ Multilingual embeddings
✅ ChromaDB vector database
✅ Hybrid search (RRF)
✅ Policy/Technical/Governance source split for Day 4 dual-track retrieval
✅ Results saved
✅ Metadata saved

📁 FILES GENERATED:
   - chroma_knowledge_base/   ← IMPORTANT for Day 4
   - knowledge_base_metadata.json
   - Day3_Hybrid_Search_Results.txt

📌 Keep these files for Day 4!
""")
print("=" * 70)

import shutil
shutil.make_archive("chroma_knowledge_base", 'zip', "chroma_knowledge_base")
print("✅ تم ضغط المجلد بنجاح!")
