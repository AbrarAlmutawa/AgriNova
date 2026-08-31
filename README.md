# AgriNova

**From Plant Sounds to Evidence-Based Agricultural Intelligence**

AgriNova is an AI-driven agricultural proof-of-concept that explores plant acoustic signals as a new source of information for crop monitoring. Instead of waiting for visible symptoms of stress, AgriNova listens to the plant itself: tomato-plant acoustic recordings are converted into Mel-spectrograms and classified (Cut / Dry) using a MobileNetV2-based CNN. The resulting prediction is connected to a Knowledge Base of technical agricultural sources and Saudi water/agriculture policy documents, turning a model prediction into a traceable, evidence-supported recommendation.

## How it works

1. **Audio input** – a tomato-plant acoustic recording is provided (required). A photo of the plant can optionally be attached, kept for historical record-keeping alongside the audio — it is not used as model input.
2. **Audio → image** – the recording is converted into a Mel-spectrogram.
3. **Classification** – a MobileNetV2-based CNN predicts the plant condition (Cut / Dry).
4. **Knowledge retrieval** – the prediction is passed to a Knowledge Base, which retrieves relevant technical and policy evidence (Gemini-powered).
5. **Evidence-based output** – the system returns Evidence, Reasoning, and a Recommendation, backed by cited sources.

Supporting features: GIS-based mapping of farm sites and a prediction history log per site, both backed by a local database of farm site metadata (location, field zone) and past predictions.

## Tech stack

- **Model:** MobileNetV2 (transfer learning), Keras
- **Backend:** FastAPI (served via Uvicorn)
- **Frontend:** Streamlit
- **Knowledge Base / reasoning:** Gemini API

## Setup

```bash
git clone https://github.com/AbrarAlmutawa/AgriNova
cd AgriNova
python3 -m venv tf_env
source tf_env/bin/activate
pip install -r requirements.txt
```

### Environment variables

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export CNN_MODEL_PATH="./CNN_Model.keras"
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

Replace `YOUR_GEMINI_API_KEY` with your own Gemini API key. `CNN_MODEL_PATH` should point to the trained `.keras` model file.

### Run

Start the backend:
```bash
python3 -m uvicorn day5_api:app --reload --port 8001
```

Start the frontend (in a separate terminal, with the virtual environment activated):
```bash
streamlit run app.py
```

## Links

- **GitHub:** https://github.com/AbrarAlmutawa/AgriNova
- **Knowledge Base:** [Google Doc](https://docs.google.com/document/d/1-AJlpwE7bubil0NSH2Ga8ry5wE2kgcC7/edit?usp=drive_link&ouid=116708446478444135879&rtpof=true&sd=true)

## Project status

AgriNova is currently a proof-of-concept. It demonstrates acoustic-based classification of Cut and Dry tomato-plant conditions; it does not yet establish real-world early detection ahead of visible stress. Site/farm data used during development is placeholder data, not real farmer data.
