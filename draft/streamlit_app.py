# -*- coding: utf-8 -*-
"""
streamlit_app.py — Day 5: Streamlit interface

Per the plan: "Show uploaded audio, spectrogram, prediction, confidence,
recommendation, and sources." This talks to the running FastAPI server
(day5_api.py) over HTTP — it does not import or duplicate any of its logic.

SETUP:
    pip install streamlit requests librosa matplotlib

USAGE:
    1. Make sure day5_api.py is already running in another terminal tab
       (uvicorn day5_api:app --reload --port 8001)
    2. In a NEW terminal tab, from the same project folder, run:
           ./tf_env/bin/python -m streamlit run streamlit_app.py
    3. Streamlit opens a browser tab automatically (usually localhost:8501)
"""

import io
import numpy as np
import requests
import streamlit as st
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="AgriNova", page_icon="🌱", layout="centered")

st.title("🌱 AgriNova — Plant Stress Detection")
st.write(
    "Upload a plant sound recording (.wav) to classify it as **Dry** or "
    "**Cut**, and get an evidence-based irrigation recommendation grounded "
    "in the AgriNova knowledge base."
)

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("FastAPI URL", value="http://localhost:8001")
    if st.button("Check API health"):
        try:
            r = requests.get(f"{api_url}/health", timeout=5)
            r.raise_for_status()
            st.success(r.json())
        except requests.exceptions.RequestException as e:
            st.error(f"API not reachable: {e}")

uploaded_file = st.file_uploader("Upload audio (.wav)", type=["wav"])

if uploaded_file is not None:
    audio_bytes = uploaded_file.getvalue()

    st.subheader("Uploaded Audio")
    st.audio(audio_bytes, format="audio/wav")

    # Spectrogram preview for the demo (visual only — a quick look at what
    # the CNN is analyzing; not required to be pixel-identical to the exact
    # array the model receives, since day5_api.py handles that separately).
    st.subheader("Spectrogram Preview")
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        fig, ax = plt.subplots(figsize=(6, 3))
        librosa.display.specshow(S_dB, sr=sr, ax=ax)
        ax.set(title="Mel Spectrogram")
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.warning(f"Could not render spectrogram preview: {e}")

    analyze = st.button("Analyze", type="primary")

    if analyze:
        with st.spinner("Classifying and retrieving evidence-based recommendation..."):
            try:
                files = {"file": (uploaded_file.name, audio_bytes, "audio/wav")}
                response = requests.post(f"{api_url}/predict", files=files, timeout=60)
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the API at {api_url}: {e}")
                st.stop()

        st.subheader("Prediction")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Classification", result["prediction"])
        with col2:
            st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")

        st.subheader("Recommendation")
        if result["llm_called"]:
            st.success("Evidence-based recommendation generated from the knowledge base.")
        else:
            st.warning("Insufficient evidence in the knowledge base for this specific case.")
        st.markdown(result["recommendation"])

        if result["technical_sources"] or result["policy_sources"]:
            st.subheader("Sources")
            for s in result["technical_sources"]:
                st.write(f"📄 {s}")
            for s in result["policy_sources"]:
                st.write(f"📜 {s}")
else:
    st.info("Upload a .wav file to get started.")
