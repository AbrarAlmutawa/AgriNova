# 🍅 AgriNova

### *From Plant Sounds to Evidence-Based Agricultural Intelligence* 🌱🔊

> What if a plant could tell you it's thirsty... before you could ever see it?

AgriNova is an AI-driven agricultural proof-of-concept that explores plant acoustic signals as a new source of information for crop monitoring. Instead of waiting for visible symptoms of stress, AgriNova **listens to the plant itself** 🎙️ — tomato-plant acoustic recordings are converted into Mel-spectrograms and classified (`Cut` / `Dry`) using a MobileNetV2-based CNN. The prediction is then connected to a Knowledge Base of technical agricultural sources and Saudi water/agriculture policy documents, turning a raw model prediction into a traceable, evidence-backed recommendation.

---

## 🔗 Links
- 🎬 **Demo:** https://youtu.be/spjAZUZorMM
- 💻 **GitHub:** https://github.com/AbrarAlmutawa/AgriNova
- 📚 **Knowledge Base:** [Google Doc](https://docs.google.com/document/d/1-AJlpwE7bubil0NSH2Ga8ry5wE2kgcC7/edit?usp=drive_link&ouid=116708446478444135879&rtpof=true&sd=true)

---

## 🚀 How it works

```
🎙️  Audio Input  →  🖼️  Mel-Spectrogram  →  🧠  CNN Classification  →  📚  Knowledge Retrieval  →  ✅  Evidence-Based Recommendation
```

1. **🎙️ Audio input** — a tomato-plant acoustic recording is provided (required). A 📷 photo of the plant can optionally be attached too, kept for historical record-keeping alongside the audio — it doesn't feed the model, it's just for the farmer's own record.
2. **🖼️ Audio → image** — the recording is converted into a Mel-spectrogram.
3. **🧠 Classification** — a MobileNetV2-based CNN predicts the plant condition (`Cut` or `Dry`).
4. **📚 Knowledge retrieval** — the prediction is passed to a Knowledge Base (powered by Gemini), which retrieves relevant technical and policy evidence.
5. **✅ Evidence-based output** — the system returns **Evidence**, **Reasoning**, and a **Recommendation**, backed by cited sources.

**Bonus features:** 🗺️ GIS-based mapping of farm sites + 📈 prediction history log per site, both backed by a local database of farm site metadata.

---

## 🛠️ Tech stack

| Layer | Tech |
|---|---|
| 🧠 Model | MobileNetV2 (transfer learning), Keras |
| ⚙️ Backend | FastAPI (served via Uvicorn) |
| 🎨 Frontend | Streamlit |
| 📚 Knowledge Base / reasoning | Gemini API |

---

## 📦 Setup

```bash
git clone https://github.com/AbrarAlmutawa/AgriNova
cd AgriNova
python3 -m venv tf_env
source tf_env/bin/activate
pip install -r requirements.txt
```

### 🔑 Environment variables

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export CNN_MODEL_PATH="./CNN_Model.keras"
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

> Swap in your own Gemini API key, and make sure `CNN_MODEL_PATH` points to the trained `.keras` model file. 🔐

### ▶️ Run it

Fire up the backend:
```bash
python3 -m uvicorn day5_api:app --reload --port 8001
```

Then, in a separate terminal (venv still activated), launch the frontend:
```bash
streamlit run app.py
```

You're in! 🎉

---

## 👩‍💻 Team

- Aisha Sami Al Abdulqader
- Sara Mousa Almousa
- Jana Mustafa Alhumaidan
- Abrar Hassan Almutawa

---


## 🌾 Project status

AgriNova is currently a **proof-of-concept** 🧪. It demonstrates acoustic-based classification of `Cut` and `Dry` tomato-plant conditions — it doesn't yet claim real-world early detection ahead of visible stress. Farm/site data used during development is placeholder data, not real farmer data.

*Don't wait until the plant looks stressed. Listen before you see it.* 🍅🎧
