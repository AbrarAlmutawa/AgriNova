# -*- coding: utf-8 -*-
"""
Day4_reranker_recommendation.py — Day 4: Reranker + Evidence-Based
Recommendation Engine (LOCAL VERSION)

WHAT THIS BUILDS ON TOP OF DAY 3
---------------------------------
Day 3 already gave you: PDF extraction (PyMuPDF), chunking, BM25, ChromaDB
vector search, hybrid RRF search, and a policy/technical/governance source
split with policy_search() / technical_search() / build_evidence_context().

Day 4 adds the three things the plan calls for that Day 3 doesn't do yet:

  1. RERANKER
     technical_search()/policy_search() return the top-10 candidates by
     embedding-distance, which is a decent first pass but not precise
     enough — it lets in things like bibliography lists or "Key Resources
     Table" boilerplate that are topically *near* the query but not
     actually useful evidence. A cross-encoder reranker re-scores those
     10 candidates specifically against the query text (not just vector
     distance) and keeps only the genuinely relevant top 3 (technical) /
     top 2 (policy).

  2. RECOMMENDATION PROMPT
     build_recommendation_prompt() assembles the surviving, reranked
     evidence into the Evidence -> Reasoning -> Recommendation -> Sources
     structure your plan specifies (Day 4, step 3), then get_recommendation()
     sends that to Gemini 2.5 Flash (free tier) to write the final
     farmer-facing answer.

  3. HALLUCINATION GUARD
     This happens BEFORE the LLM is ever called, not after. If nothing
     clears the reranker's min_score threshold, get_recommendation() returns
     an "insufficient evidence" message directly and never calls the LLM at
     all. The LLM is never handed an empty or weak context and is never in
     a position to invent an answer — this is what your plan means by
     "prevent hallucination" (Day 4, step 4), implemented as a hard gate
     rather than a hope that the LLM polices itself.

Also included: an 18-question RAG test harness (irrigation, drought, two
Arabic queries, Saudi policy, deliberately irrelevant questions, and one
meta/self-referential question) that exercises the full pipeline and logs
what evidence/sources came back for each — this is Day 4 deliverable
"15-20 RAG test cases" from your plan.

WHAT THIS SCRIPT DOES NOT REDO
--------------------------------
It does NOT re-extract PDFs into a fresh ChromaDB collection and does NOT
re-run the embedding model over your documents. It reconnects to the
persistent ChromaDB collection Day 3 already built ("agri_knowledge_base"
in ./chroma_knowledge_base via client.get_collection, not create_collection).

The only thing rebuilt locally is the BM25 index, since BM25 isn't
persisted by ChromaDB. It is rebuilt from the chunks pulled back out of
Chroma via collection.get() — NOT by re-reading or re-chunking the PDFs.
extract_pdf_pages() and create_chunks() from Day 3 are not re-run at all;
this is pure tokenization over text Chroma already has, so it costs
milliseconds and touches no PDF files.

SETUP (on top of Day 3's requirements)
----------------------------------------
    pip install sentence-transformers requests
    # (sentence-transformers should already be installed from Day 3;
    #  the cross-encoder reranker model downloads automatically on first run)

    Set your Gemini API key as an environment variable before running:
        export GEMINI_API_KEY="your-key-here"
    Get a free key (no card required) from Google AI Studio.

USAGE
-----
    python Day4_reranker_recommendation.py --chroma_dir ./chroma_knowledge_base
"""

import os
import re
import sys
import json
import argparse
import numpy as np
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from datetime import datetime

try:
    from pyarabic.araby import strip_tashkeel, normalize_hamza, normalize_alef
    HAVE_PYARABIC = True
except ImportError:
    HAVE_PYARABIC = False

print("=" * 70)
print("DAY 4 — RERANKER + EVIDENCE-BASED RECOMMENDATION (LOCAL)")
print("=" * 70)

# ============================================================
# 0. ARGS
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--chroma_dir", default="./chroma_knowledge_base",
                     help="Path to the persistent ChromaDB folder Day 3 created")
args, _ = parser.parse_known_args()

CHROMA_DIR = args.chroma_dir
COLLECTION_NAME = "agri_knowledge_base"

# ============================================================
# 1. RECONNECT TO EXISTING CHROMADB COLLECTION  (get, not create)
# ============================================================
#
# No PDF re-parsing here at all. Day 3 already read every PDF, chunked it,
# and stored (id, text, source, page) for every chunk inside this Chroma
# collection. extract_pdf_pages() and create_chunks() from Day 3 are NOT
# re-run — collection.get() below pulls the exact same chunk objects back
# out, so there's zero duplicate work with what you already ran.

print("\nReconnecting to existing ChromaDB collection...")

client = chromadb.PersistentClient(path=CHROMA_DIR)
try:
    collection = client.get_collection(COLLECTION_NAME)
except Exception as e:
    raise RuntimeError(
        f"Could not find collection '{COLLECTION_NAME}' in {CHROMA_DIR}.\n"
        f"Run Day 3 first, or check --chroma_dir. Original error: {e}"
    )

print(f"Connected. Chunks stored in Chroma: {collection.count()}")

# ============================================================
# 2. PULL CHUNKS BACK OUT OF CHROMA + REBUILD BM25 ONLY
# ============================================================
#
# BM25 itself is the one piece Chroma doesn't store (it's a keyword index,
# not a vector one), so it's the only thing rebuilt here — from the chunks
# just retrieved above, not from the PDFs. That's tokenization only
# (milliseconds), no PDF I/O, no embedding model involved.

raw = collection.get(include=["documents", "metadatas"])
chunks = [
    {"id": raw["ids"][i], "text": raw["documents"][i],
     "source": raw["metadatas"][i]["source"], "page": raw["metadatas"][i]["page"]}
    for i in range(len(raw["ids"]))
]
chunk_texts = [c["text"] for c in chunks]
pdf_files = sorted(set(c["source"] for c in chunks))

def normalize_for_bm25(text):
    text = text.lower()
    if HAVE_PYARABIC:
        text = strip_tashkeel(text)
        text = normalize_hamza(text)
        text = normalize_alef(text)
    else:
        text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
        text = re.sub(r"[إأآا]", "ا", text)
        text = re.sub(r"ة", "ه", text)
    return text

tokenized_chunks = [normalize_for_bm25(t).split() for t in chunk_texts]
bm25 = BM25Okapi(tokenized_chunks)

print(f"BM25 rebuilt from {len(chunks)} chunks already stored in Chroma (no PDF re-read).")

embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
print("Embedding model loaded (for query encoding only).")

# ============================================================
# 3. SOURCE SPLIT  (same lists as Day 3 — adjust if filenames differ)
# ============================================================

POLICY_SOURCES = [s for s in pdf_files if any(k in s for k in
    ["الاستراتيجية", "قاب", "SaudiWaterResourcesCode"])]
TECHNICAL_SOURCES = [s for s in pdf_files if any(k in s for k in
    ["FAO_Crop_Water_Response", "FAO_Irrigation_Manual",
     "MEWA_Tomato_Irrigation_Guide", "Khait_Plant_Sounds"])]

# ============================================================
# 4. SEARCH FUNCTIONS  (bm25/vector/hybrid — same as Day 3)
# ============================================================

def bm25_search(query, top_k=5):
    scores = bm25.get_scores(normalize_for_bm25(query).split())
    top = np.argsort(scores)[::-1][:top_k]
    results = []
    for r, idx in enumerate(top, 1):
        if scores[idx] > 0:
            results.append({"rank": r, "text": chunks[idx]["text"],
                             "source": chunks[idx]["source"], "page": chunks[idx]["page"],
                             "score": float(scores[idx])})
    return results

def vector_search_scoped(query, top_k, sources):
    if not sources:
        return []
    q_emb = embedding_model.encode([query], normalize_embeddings=True)[0]
    res = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=top_k,
        where={"source": {"$in": sources}},
        include=["documents", "metadatas", "distances"],
    )
    results = []
    if not res["documents"][0]:
        return results
    for i in range(len(res["documents"][0])):
        results.append({
            "rank": i + 1,
            "text": res["documents"][0][i],
            "source": res["metadatas"][0][i]["source"],
            "page": res["metadatas"][0][i]["page"],
            "distance": res["distances"][0][i],
        })
    return results

def technical_search(query, top_k=10):
    return vector_search_scoped(query, top_k, TECHNICAL_SOURCES)

def policy_search(query, top_k=10):
    return vector_search_scoped(query, top_k, POLICY_SOURCES)

# ============================================================
# 5. NEW — RERANKER (Day 4 core piece #1)
# ============================================================

print("\nLoading cross-encoder reranker (multilingual, Arabic + English)...")
reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
print("Reranker ready.")

# Tune these against your own real queries before demo day.
TECHNICAL_MIN_SCORE = 0.0   # cross-encoder logit threshold for technical evidence
POLICY_MIN_SCORE = 0.5      # stricter — a wrong regulatory citation is worse than a wrong agronomic one

def rerank(query, candidates, top_k, min_score):
    """Re-score `candidates` against `query` with the cross-encoder and keep
    only the top_k that clear min_score. Returns [] if nothing clears the bar
    — this empty list is what powers the hallucination guard downstream."""
    if not candidates:
        return []
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)
    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    kept = [(c, s) for c, s in scored if s >= min_score][:top_k]
    results = []
    for rank, (c, s) in enumerate(kept, 1):
        results.append({"rank": rank, "text": c["text"], "source": c["source"],
                         "page": c["page"], "rerank_score": float(s)})
    return results

def reranked_technical_search(query, candidate_k=10, top_k=3):
    candidates = technical_search(query, top_k=candidate_k)
    return rerank(query, candidates, top_k=top_k, min_score=TECHNICAL_MIN_SCORE)

def reranked_policy_search(query, candidate_k=10, top_k=2):
    candidates = policy_search(query, top_k=candidate_k)
    return rerank(query, candidates, top_k=top_k, min_score=POLICY_MIN_SCORE)

# ============================================================
# 6. NEW — RECOMMENDATION PROMPT (Day 4 core piece #2)
# ============================================================

def build_recommendation_prompt(query, technical_evidence, policy_evidence):
    """Evidence -> Reasoning -> Recommendation -> Sources, per the plan."""
    evidence_block = ""
    for e in technical_evidence:
        evidence_block += f"- [TECHNICAL | {e['source']}, p.{e['page']}] {e['text'][:500]}\n"
    for e in policy_evidence:
        evidence_block += f"- [REGULATORY | {e['source']}, p.{e['page']}] {e['text'][:500]}\n"

    prompt = f"""You are an agricultural advisory assistant. Answer ONLY using the evidence
below. Do not add information that is not supported by the evidence.

FARMER QUESTION:
{query}

RETRIEVED EVIDENCE:
{evidence_block}

Respond in exactly this structure:

Evidence: Summarize what the retrieved sources actually say, in your own words.
Reasoning: Explain how this evidence applies to the farmer's specific question.
Recommendation: Give a clear, actionable recommendation grounded strictly in the evidence above.
Sources: List each source used as "filename, page X" followed by a short direct quote
(under 25 words) copied verbatim from that source's evidence text above, showing exactly
what backs the recommendation. Format each as:
  - filename, page X: "exact quoted excerpt from the evidence"
Only quote text that actually appears in the RETRIEVED EVIDENCE above — never invent or
paraphrase text inside the quotation marks.

If the evidence above is insufficient to answer confidently, say so explicitly in the
Recommendation section instead of guessing.
"""
    return prompt

# ============================================================
# 7. LLM CALL  (Gemini 2.5 Flash — free tier, no card required)
# ============================================================
#
# NOTE: unlike Cohere's Command R, Gemini has no native "documents" /
# citation parameter. That's not a problem here — your reranker + min_score
# guard already verified relevance before this function is even called, and
# build_recommendation_prompt() already embeds the evidence with explicit
# [source, page] tags and instructs the model to cite them back in the
# Sources section. Gemini just has to follow that structure, not invent it.

GEMINI_MODEL = "gemini-3.6-flash"  # 2.5-flash deprecated for new API keys as of Aug 2026; swap to another current Flash model if this also changes

def call_llm(prompt, technical_evidence, policy_evidence):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return ("[NO LLM CALL MADE — set GEMINI_API_KEY to enable this]\n"
                "Prompt that would have been sent:\n" + prompt)

    import requests
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    try:
        resp = requests.post(
            url,
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            json={
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2},
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return f"[LLM CALL RETURNED NO CANDIDATES: {json.dumps(data)[:500]}]"
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        return text if text else json.dumps(data)[:1000]
    except requests.exceptions.HTTPError as e:
        return f"[LLM CALL FAILED: HTTP {e.response.status_code} — {e.response.text[:300]}]"
    except Exception as e:
        return f"[LLM CALL FAILED: {e}]"

# ============================================================
# 8. NEW — FULL PIPELINE WITH HALLUCINATION GUARD (Day 4 core piece #3)
# ============================================================

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "Evidence: No sources in the knowledge base cleared the relevance threshold "
    "for this question.\n"
    "Reasoning: Answering without adequate supporting evidence risks giving "
    "incorrect agricultural or regulatory advice.\n"
    "Recommendation: The available sources are insufficient to answer this "
    "question confidently. Please consult a local agricultural extension "
    "office or the relevant Saudi authority directly.\n"
    "Sources: None met the relevance threshold."
)

def get_recommendation(query):
    technical_evidence = reranked_technical_search(query)
    policy_evidence = reranked_policy_search(query)

    # HALLUCINATION GUARD — checked before any LLM call.
    if not technical_evidence and not policy_evidence:
        return {
            "query": query,
            "answer": INSUFFICIENT_EVIDENCE_MESSAGE,
            "technical_evidence": [],
            "policy_evidence": [],
            "llm_called": False,
        }

    prompt = build_recommendation_prompt(query, technical_evidence, policy_evidence)
    answer = call_llm(prompt, technical_evidence, policy_evidence)

    return {
        "query": query,
        "answer": answer,
        "technical_evidence": technical_evidence,
        "policy_evidence": policy_evidence,
        "llm_called": True,
    }

# ============================================================
# 9. TEST HARNESS — 18 QUESTIONS (Day 4 deliverable: 15-20 RAG test cases)
# ============================================================

test_questions = [
    # Irrigation
    "What irrigation practices are recommended for tomato plants?",
    "How much water do tomato crops need during flowering?",
    "What is deficit irrigation and when should it be used?",
    # Drought / plant stress
    "What are the early signs of drought stress in plants?",
    "How does water stress affect crop yield?",
    "Can plant sounds indicate drought stress?",
    # General agriculture / water management
    "How can water resources be managed efficiently in agriculture?",
    "What agricultural practices support sustainable water use?",
    "What is the FAO crop water requirement method?",
    # Saudi policy / regulation
    "What does Saudi water policy say about agricultural water use?",
    "What are the goals of the Saudi National Water Strategy 2030?",
    "What regulations exist under the Saudi Water Resources Code?",
    "What does Saudi GAP require for irrigation practices?",
    # Arabic queries
    "ما هي ممارسات الري الموصى بها لنباتات الطماطم؟",
    "ما هي علامات الإجهاد المائي في النباتات؟",
    # Deliberately irrelevant (should trigger the hallucination guard)
    "What is the best fertilizer brand to buy in 2026?",
    "What is the capital of France?",
    # Meta / self-referential (should NOT get a fabricated answer about internals)
    "What model are you using to generate this recommendation?",
]

print("\n" + "=" * 70)
print("RUNNING DAY 4 RAG TEST SUITE")
print("=" * 70)

all_results = []
for i, q in enumerate(test_questions, 1):
    print(f"\n[{i}/{len(test_questions)}] {q}")
    result = get_recommendation(q)
    all_results.append(result)
    n_tech = len(result["technical_evidence"])
    n_pol = len(result["policy_evidence"])
    status = "ANSWERED" if result["llm_called"] else "INSUFFICIENT EVIDENCE (guard triggered)"
    print(f"   -> {status} | technical={n_tech} policy={n_pol}")

# ============================================================
# 10. SAVE RESULTS
# ============================================================

with open("Day4_RAG_Test_Results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

with open("Day4_RAG_Test_Results.txt", "w", encoding="utf-8") as f:
    f.write("DAY 4 — RAG TEST RESULTS\n")
    f.write(f"Generated: {datetime.now().isoformat()}\n")
    f.write("=" * 70 + "\n\n")
    for r in all_results:
        f.write(f"QUERY: {r['query']}\n")
        f.write(f"LLM CALLED: {r['llm_called']}\n")
        f.write(f"TECHNICAL EVIDENCE ({len(r['technical_evidence'])}):\n")
        for e in r["technical_evidence"]:
            f.write(f"  - [{e['source']}, p.{e['page']}] score={e['rerank_score']:.3f}\n")
        f.write(f"POLICY EVIDENCE ({len(r['policy_evidence'])}):\n")
        for e in r["policy_evidence"]:
            f.write(f"  - [{e['source']}, p.{e['page']}] score={e['rerank_score']:.3f}\n")
        f.write(f"ANSWER:\n{r['answer']}\n")
        f.write("-" * 70 + "\n\n")

print("\n" + "=" * 70)
print("DAY 4 COMPLETE")
print("=" * 70)
print("""
Files generated:
  - Day4_RAG_Test_Results.json
  - Day4_RAG_Test_Results.txt

Next: review the .txt file — check that the two irrelevant questions
triggered the insufficient-evidence guard, and that Arabic queries pulled
the right Arabic-source evidence. Tune TECHNICAL_MIN_SCORE / POLICY_MIN_SCORE
if too much noise gets through or too much good evidence gets filtered out.
""")
