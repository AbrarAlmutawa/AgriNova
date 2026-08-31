# -*- coding: utf-8 -*-
"""
day5_api.py — Day 5: Full System Integration (FastAPI)

Target pipeline per the plan:
  Audio -> Preprocessing -> Mel Spectrogram -> CNN -> Dry/Cut ->
  Hybrid RAG -> Evidence -> Recommendation -> (Streamlit)

This file exposes POST /predict, which:
  1. Accepts an uploaded .wav file
  2. Runs preprocessing (librosa: WAV -> log-Mel spectrogram)
  3. Runs the CNN classifier -> "Dry" or "Cut" + confidence score
  4. Builds a query from that prediction and runs it through Day 4's
     full RAG pipeline (hybrid retrieval -> rerank -> hallucination
     guard -> Gemini recommendation)
  5. Returns prediction, confidence, recommendation, and sources as JSON

============================================================
CONFIRMED FROM BOTH NOTEBOOKS — preprocessing is now exact
============================================================
tomatoes.ipynb (Day 1, save_spectrogram()) confirms the exact pipeline:
  librosa.load(path, sr=None)  -> native sample rate, no resampling
  librosa.feature.melspectrogram(y, sr, n_mels=128)  -> default n_fft/hop
  librosa.power_to_db(S, ref=np.max)
  plt.figure(figsize=(3,3)); axis off; librosa.display.specshow(S_dB, sr=sr)
  plt.savefig(..., bbox_inches="tight", pad_inches=0)
  -> resized to (128,128) at train time via ImageDataGenerator(target_size=...)

_AgriNova.ipynb (Day 2, EfficientNetB0 transfer learning) confirms:
  - Model file: phase2_finetuned.keras (the fine-tuned version)
  - Input shape: (128, 128, 3), RGB
  - Class order: Cut=0, Dry=1
  - No rescaling applied (rescale=1./255 was commented out) -> raw 0-255

wav_bytes_to_spectrogram_image() below reproduces the Day 1 rendering
step exactly. The one thing still worth confirming with your teammate:
which exact training run produced phase2_finetuned.keras, since her Day 2
notebook's Phase 1 cell crashes with NameError (class_weight_dict
undefined) — make sure the saved file came from a run where that was
actually fixed.

SETUP (on top of Day 3/4 requirements):
    pip install fastapi uvicorn python-multipart librosa soundfile tensorflow matplotlib

USAGE:
    export GEMINI_API_KEY="your-key-here"
    export KMP_DUPLICATE_LIB_OK=TRUE
    uvicorn day5_api:app --reload --port 8000

    Then POST a .wav file to http://localhost:8000/predict
    (e.g. via the Streamlit app, or curl -F "file=@sample.wav" http://localhost:8000/predict)
"""

import os
import re
import io
import json
import numpy as np
import tensorflow as tf
import librosa
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from plant_store import PlantStore

try:
    from pyarabic.araby import strip_tashkeel, normalize_hamza, normalize_alef
    HAVE_PYARABIC = True
except ImportError:
    HAVE_PYARABIC = False

# ============================================================
# CONFIG
# ============================================================

CHROMA_DIR = os.environ.get("CHROMA_DIR", "./chroma_knowledge_base")
COLLECTION_NAME = "agri_knowledge_base"
MODEL_PATH = os.environ.get("CNN_MODEL_PATH", "./phase2_finetuned.keras")  # confirmed from her notebook
GEMINI_MODEL = "gemini-3.6-flash"

# Confirmed from her Day 1 notebook (tomatoes.ipynb, save_spectrogram()):
IMAGE_SIZE = (128, 128)      # matches her ImageDataGenerator target_size
CLASS_NAMES = ["Cut", "Dry"]  # confirmed: her class_indices = {'Cut': 0, 'Dry': 1}

# NOTE: her pipeline does NOT resample audio to a fixed sample rate and does
# NOT pad/trim clips to a fixed duration — librosa.load(path, sr=None) keeps
# each file's native sample rate, and the spectrogram is rendered into a
# fixed-size (3x3 inch) figure regardless of clip length. n_mels=128 is
# confirmed; n_fft/hop_length were left at librosa's defaults (2048/512).
MEL_N_MELS = 128

# ============================================================
# AUDIO PREPROCESSING  (Day 1 equivalent)
# ============================================================

def wav_bytes_to_spectrogram_image(audio_bytes: bytes) -> np.ndarray:
    """WAV bytes -> (128, 128, 3) RGB array, matching her Day 1 save_spectrogram()
    function exactly (confirmed from tomatoes.ipynb, cell 3):

        y, sr = librosa.load(audio_path, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        plt.figure(figsize=(3, 3)); plt.axis("off")
        librosa.display.specshow(S_dB, sr=sr)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)

    Her code has no augmentation branch active for live inference (that only
    applies during her Day 1 training-set generation), so this reproduces
    only the base path. The rendered PNG is then resized to IMAGE_SIZE the
    same way her ImageDataGenerator(target_size=(128,128)) did at train time.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa.display

    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)  # native sample rate, no resample

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=MEL_N_MELS)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig = plt.figure(figsize=(3, 3))
    plt.axis("off")
    librosa.display.specshow(S_dB, sr=sr)  # no cmap arg -> matplotlib default ('viridis')

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    # Resize to IMAGE_SIZE the same way flow_from_directory(target_size=...) did.
    img = tf.keras.utils.load_img(buf, target_size=IMAGE_SIZE)
    arr = tf.keras.utils.img_to_array(img)  # (128, 128, 3)

    # Her rescale=1./255 was commented out -> raw 0-255 pixel values, unnormalized.
    return arr.astype(np.float32)

# ============================================================
# CNN MODEL  (Day 2 equivalent — Keras)
# ============================================================
# Unlike PyTorch, a Keras .h5/.keras file saved via model.save() already
# contains the full architecture, so there's no separate class to define
# here. load_model() reconstructs everything from the file.

def load_cnn_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Get phase2_finetuned.keras from "
            f"your teammate and place it in the project folder, or set "
            f"CNN_MODEL_PATH to point at it."
        )
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Loaded CNN model from {MODEL_PATH}")
    print(f"  Expected input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    # Sanity check: should read (None, 128, 128, 3) per her Day 2 notebook.
    # If it doesn't, IMAGE_SIZE above needs to be updated to match.
    return model

def predict_dry_cut(image_array: np.ndarray, model):
    x = np.expand_dims(image_array, axis=0)  # add batch dim -> (1, 128, 128, 3)
    output = model.predict(x, verbose=0)[0]

    # TODO(teammate): confirm whether her final layer is softmax (2 outputs
    # summing to 1) or sigmoid (1 output, binary). This handles both:
    if output.shape[-1] == 1:
        prob_cut = float(output[0])
        pred_idx = 1 if prob_cut >= 0.5 else 0
        confidence = prob_cut if pred_idx == 1 else 1 - prob_cut
    else:
        pred_idx = int(np.argmax(output))
        confidence = float(output[pred_idx])

    return CLASS_NAMES[pred_idx], confidence

# ============================================================
# RAG PIPELINE  (Day 3 + Day 4, reconnected)
# ============================================================

print("Connecting to knowledge base...")
client = chromadb.PersistentClient(path=CHROMA_DIR)
try:
    collection = client.get_collection(COLLECTION_NAME)
except Exception as e:
    raise RuntimeError(f"Could not find collection '{COLLECTION_NAME}' in {CHROMA_DIR}. Run Day 3 first. {e}")

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

embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

POLICY_SOURCES = [s for s in pdf_files if any(k in s for k in
    ["الاستراتيجية", "قاب", "SaudiWaterResourcesCode"])]
TECHNICAL_SOURCES = [s for s in pdf_files if any(k in s for k in
    ["FAO_Crop_Water_Response", "FAO_Irrigation_Manual",
     "MEWA_Tomato_Irrigation_Guide", "Khait_Plant_Sounds"])]

TECHNICAL_MIN_SCORE = 0.0
POLICY_MIN_SCORE = 0.5

def vector_search_scoped(query, top_k, sources):
    if not sources:
        return []
    q_emb = embedding_model.encode([query], normalize_embeddings=True)[0]
    res = collection.query(query_embeddings=[q_emb.tolist()], n_results=top_k,
                            where={"source": {"$in": sources}},
                            include=["documents", "metadatas", "distances"])
    if not res["documents"][0]:
        return []
    return [{"rank": i + 1, "text": res["documents"][0][i],
              "source": res["metadatas"][0][i]["source"],
              "page": res["metadatas"][0][i]["page"]}
             for i in range(len(res["documents"][0]))]

def rerank(query, candidates, top_k, min_score):
    if not candidates:
        return []
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)
    scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    kept = [(c, s) for c, s in scored if s >= min_score][:top_k]
    return [{"rank": r + 1, "text": c["text"], "source": c["source"],
              "page": c["page"], "rerank_score": float(s)}
             for r, (c, s) in enumerate(kept)]

def reranked_technical_search(query, candidate_k=10, top_k=3):
    return rerank(query, vector_search_scoped(query, candidate_k, TECHNICAL_SOURCES), top_k, TECHNICAL_MIN_SCORE)

def reranked_policy_search(query, candidate_k=10, top_k=2):
    return rerank(query, vector_search_scoped(query, candidate_k, POLICY_SOURCES), top_k, POLICY_MIN_SCORE)

def build_recommendation_prompt(query, technical_evidence, policy_evidence):
    evidence_block = ""
    for e in technical_evidence:
        evidence_block += f"- [TECHNICAL | {e['source']}, p.{e['page']}] {e['text'][:500]}\n"
    for e in policy_evidence:
        evidence_block += f"- [REGULATORY | {e['source']}, p.{e['page']}] {e['text'][:500]}\n"

    return f"""You are an agricultural advisory assistant. Answer ONLY using the evidence
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
(under 25 words) copied verbatim from that source's evidence text above. Format each as:
  - filename, page X: "exact quoted excerpt from the evidence"
Only quote text that actually appears in the RETRIEVED EVIDENCE above.

If the evidence above is insufficient to answer confidently, say so explicitly in the
Recommendation section instead of guessing.
"""

def call_gemini(prompt):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "[NO LLM CALL MADE — GEMINI_API_KEY not set]"
    import requests
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    try:
        resp = requests.post(url,
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            json={"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                  "generationConfig": {"temperature": 0.2}},
            timeout=30)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return f"[LLM RETURNED NO CANDIDATES: {json.dumps(data)[:300]}]"
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts)
    except Exception as e:
        return f"[LLM CALL FAILED: {e}]"

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "Evidence: No sources in the knowledge base cleared the relevance threshold for this question.\n"
    "Reasoning: Answering without adequate supporting evidence risks giving incorrect advice.\n"
    "Recommendation: The available sources are insufficient to answer confidently. "
    "Please consult a local agricultural extension office or the relevant Saudi authority directly.\n"
    "Sources: None met the relevance threshold."
)

def get_recommendation(query):
    technical_evidence = reranked_technical_search(query)
    policy_evidence = reranked_policy_search(query)
    if not technical_evidence and not policy_evidence:
        return {"answer": INSUFFICIENT_EVIDENCE_MESSAGE, "technical_evidence": [],
                "policy_evidence": [], "llm_called": False}
    prompt = build_recommendation_prompt(query, technical_evidence, policy_evidence)
    answer = call_gemini(prompt)
    return {"answer": answer, "technical_evidence": technical_evidence,
            "policy_evidence": policy_evidence, "llm_called": True}

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(title="AgriNova API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Streamlit runs on a different port locally; fine for a hackathon demo
    allow_methods=["*"],
    allow_headers=["*"],
)

cnn_model = load_cnn_model()
plant_store = PlantStore()

class PredictResponse(BaseModel):
    plant_id: str
    prediction: str
    confidence: float
    recommendation: str
    technical_sources: list
    policy_sources: list
    llm_called: bool

class PlantCreateRequest(BaseModel):
    name: str
    latitude: float
    longitude: float
    field_zone: str = None

class PlantResponse(BaseModel):
    plant_id: str
    name: str
    latitude: float
    longitude: float
    field_zone: str = None
    created_at: str
    photo_filename: str = None

# ============================================================
# PLANT REGISTRY + GIS ENDPOINTS
# ============================================================

@app.post("/plants", response_model=PlantResponse)
async def register_plant(plant: PlantCreateRequest):
    plant_id = plant_store.create_plant(
        plant.name, plant.latitude, plant.longitude, plant.field_zone
    )
    # Read back the full stored row so created_at (and any other server-set
    # fields) are populated correctly, rather than hand-assembling the
    # response from the incoming request alone.
    return plant_store.get_plant(plant_id)

@app.get("/plants", response_model=list[PlantResponse])
async def list_plants():
    return plant_store.list_plants()

@app.get("/plants/{plant_id}")
async def get_plant(plant_id: str):
    plant = plant_store.get_plant(plant_id)
    if not plant:
        raise HTTPException(status_code=404, detail="Unknown plant_id")
    return plant

@app.get("/plants/{plant_id}/history")
async def plant_history(plant_id: str, limit: int = 50):
    if not plant_store.plant_exists(plant_id):
        raise HTTPException(status_code=404, detail="Unknown plant_id")
    return plant_store.get_history(plant_id, limit=limit)

@app.post("/plants/{plant_id}/photo")
async def upload_plant_photo(plant_id: str, file: UploadFile = File(...)):
    if not plant_store.plant_exists(plant_id):
        raise HTTPException(status_code=404, detail="Unknown plant_id")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")
    photo_bytes = await file.read()
    stored_filename = plant_store.set_plant_photo(plant_id, photo_bytes, file.filename)
    return {"plant_id": plant_id, "photo_filename": stored_filename}

@app.get("/plants/{plant_id}/photo")
async def get_plant_photo(plant_id: str):
    path = plant_store.get_photo_path(plant_id)
    if not path:
        raise HTTPException(status_code=404, detail="No photo uploaded for this plant")
    return FileResponse(path)

# ============================================================
# PREDICT (now plant-aware)
# ============================================================

@app.post("/predict", response_model=PredictResponse)
async def predict(plant_id: str = Form(...), file: UploadFile = File(...)):
    if not plant_store.plant_exists(plant_id):
        raise HTTPException(
            status_code=404,
            detail=f"Unknown plant_id '{plant_id}'. Register the plant first via POST /plants.",
        )

    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Please upload a .wav file")

    audio_bytes = await file.read()
    try:
        image_array = wav_bytes_to_spectrogram_image(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process audio: {e}")

    prediction, confidence = predict_dry_cut(image_array, cnn_model)

    # Pull recent history for this plant so the RAG query can reflect a
    # trend (e.g. repeated "Dry" readings) rather than just this one clip.
    recent = plant_store.recent_predictions(plant_id, n=3)
    trend_note = ""
    if len(recent) >= 2 and all(p == prediction for p in recent):
        trend_note = (
            f" This is the {len(recent)}th consecutive '{prediction}' reading "
            f"for this plant, suggesting a persistent rather than one-off condition."
        )

    query = (
        f"What irrigation and water management recommendations apply to a tomato plant "
        f"showing signs of '{prediction}' stress based on plant sound analysis?{trend_note}"
    )

    rec = get_recommendation(query)

    technical_sources = [f"{e['source']}, p.{e['page']}" for e in rec["technical_evidence"]]
    policy_sources = [f"{e['source']}, p.{e['page']}" for e in rec["policy_evidence"]]

    plant_store.log_reading(
        plant_id=plant_id,
        prediction=prediction,
        confidence=round(confidence, 4),
        recommendation=rec["answer"],
        technical_sources=technical_sources,
        policy_sources=policy_sources,
        llm_called=rec["llm_called"],
    )

    return PredictResponse(
        plant_id=plant_id,
        prediction=prediction,
        confidence=round(confidence, 4),
        recommendation=rec["answer"],
        technical_sources=technical_sources,
        policy_sources=policy_sources,
        llm_called=rec["llm_called"],
    )

@app.get("/health")
async def health():
    return {"status": "ok", "chunks_loaded": len(chunks), "model_loaded": os.path.exists(MODEL_PATH)}