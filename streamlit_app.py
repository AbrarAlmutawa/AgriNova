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
import pandas as pd
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

    st.divider()
    st.header("Plant")

    # Fetch registered plants for the dropdown.
    try:
        plants = requests.get(f"{api_url}/plants", timeout=5).json()
    except requests.exceptions.RequestException:
        plants = []

    plant_labels = {f"{p['name']} ({p['plant_id']})": p["plant_id"] for p in plants}
    selected_label = st.selectbox(
        "Select existing plant", ["— none —"] + list(plant_labels.keys())
    )
    selected_plant_id = plant_labels.get(selected_label)

    with st.expander("Register a new plant"):
        new_name = st.text_input("Plant / sensor name", key="new_plant_name")
        new_lat = st.number_input("Latitude", value=26.4207, format="%.6f")
        new_lon = st.number_input("Longitude", value=50.0888, format="%.6f")
        new_zone = st.text_input("Field zone (optional)", key="new_plant_zone")
        new_photo = st.file_uploader(
            "Photo (optional)", type=["jpg", "jpeg", "png"], key="new_plant_photo"
        )
        if st.button("Register plant"):
            if not new_name:
                st.warning("Give the plant a name first.")
            else:
                try:
                    resp = requests.post(
                        f"{api_url}/plants",
                        json={
                            "name": new_name,
                            "latitude": new_lat,
                            "longitude": new_lon,
                            "field_zone": new_zone or None,
                        },
                        timeout=5,
                    )
                    resp.raise_for_status()
                    new_plant_id = resp.json()["plant_id"]

                    if new_photo is not None:
                        photo_files = {"file": (new_photo.name, new_photo.getvalue(), new_photo.type)}
                        photo_resp = requests.post(
                            f"{api_url}/plants/{new_plant_id}/photo", files=photo_files, timeout=10
                        )
                        photo_resp.raise_for_status()

                    st.success(f"Registered: {new_plant_id}")
                    st.rerun()
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not register plant: {e}")

# ---------------- GIS map of registered plants ----------------
if plants:
    st.subheader("🗺️ Registered Plants (GIS)")
    map_df = pd.DataFrame(plants)[["latitude", "longitude"]]
    st.map(map_df, size=20)
    with st.expander("Plant details"):
        st.dataframe(pd.DataFrame(plants)[["plant_id", "name", "field_zone", "latitude", "longitude"]])

# ---------------- Selected plant photo ----------------
if selected_plant_id:
    st.subheader(f"📷 {selected_label}")
    col_photo, col_upload = st.columns([1, 1])
    with col_photo:
        photo_resp = requests.get(f"{api_url}/plants/{selected_plant_id}/photo", timeout=5)
        if photo_resp.status_code == 200:
            st.image(photo_resp.content, use_container_width=True)
        else:
            st.caption("No photo uploaded yet for this plant.")
    with col_upload:
        replace_photo = st.file_uploader(
            "Upload / replace photo", type=["jpg", "jpeg", "png"], key="replace_plant_photo"
        )
        if replace_photo is not None and st.button("Save photo"):
            try:
                files = {"file": (replace_photo.name, replace_photo.getvalue(), replace_photo.type)}
                r = requests.post(f"{api_url}/plants/{selected_plant_id}/photo", files=files, timeout=10)
                r.raise_for_status()
                st.success("Photo saved.")
                st.rerun()
            except requests.exceptions.RequestException as e:
                st.error(f"Could not upload photo: {e}")

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

    analyze = st.button("Analyze", type="primary", disabled=not selected_plant_id)
    if not selected_plant_id:
        st.caption("Select or register a plant in the sidebar before analyzing.")

    if analyze:
        with st.spinner("Classifying and retrieving evidence-based recommendation..."):
            try:
                files = {"file": (uploaded_file.name, audio_bytes, "audio/wav")}
                data = {"plant_id": selected_plant_id}
                response = requests.post(
                    f"{api_url}/predict", files=files, data=data, timeout=60
                )
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

# ---------------- Plant history ----------------
if selected_plant_id:
    st.divider()
    st.subheader(f"📈 History — {selected_label}")
    try:
        hist = requests.get(f"{api_url}/plants/{selected_plant_id}/history", timeout=5).json()
    except requests.exceptions.RequestException as e:
        hist = []
        st.warning(f"Could not load history: {e}")

    if hist:
        hist_df = pd.DataFrame(hist)[["timestamp", "prediction", "confidence"]]
        hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])
        hist_df = hist_df.sort_values("timestamp")
        st.dataframe(hist_df.iloc[::-1], use_container_width=True)
    else:
        st.caption("No readings logged for this plant yet.")

if uploaded_file is None:
    st.info("Upload a .wav file to get started.")