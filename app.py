import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import config
import pandas as pd

# Load model
model = tf.keras.models.load_model(
    config.MODEL_PATH + "heart_model.h5",
    compile=False
)
st.title("❤️ Heart Sound Disease Detection")

uploaded_file = st.file_uploader("Upload Heart Sound (.wav)", type=["wav"])


# =========================
# FIX LENGTH
# =========================
def fix_length(audio, target_length):
    if len(audio) > target_length:
        return audio[:target_length]
    elif len(audio) < target_length:
        return np.pad(audio, (0, target_length - len(audio)))
    return audio


# =========================
# SPECTROGRAM
# =========================
def audio_to_spectrogram(audio):
    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=config.SAMPLE_RATE,
        n_mels=config.N_MELS
    )
    
    spec_db = librosa.power_to_db(spec, ref=np.max)
    return spec_db


# =========================
# SEGMENT FUNCTION
# =========================
def segment_audio(audio, segment_length):
    segments = []
    
    for start in range(0, len(audio), segment_length):
        end = start + segment_length
        segment = audio[start:end]
        
        if len(segment) == segment_length:
            segments.append(segment)
    
    return segments


# =========================
# UI PROCESS
# =========================
if uploaded_file is not None:
    
    st.audio(uploaded_file)
    
    # Load audio
    y, sr = librosa.load(uploaded_file, sr=config.SAMPLE_RATE)
    
    # y = fix_length(y, config.SAMPLES)
    
    # =========================
    # WAVEFORM
    # =========================
    st.subheader("📈 Waveform")
    
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    st.pyplot(fig)
    
    # =========================
    # SPECTROGRAM
    # =========================
    st.subheader("🎧 Spectrogram")
    
    spec = audio_to_spectrogram(y)
    
    fig2, ax2 = plt.subplots()
    img = librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='mel', ax=ax2)
    fig2.colorbar(img, ax=ax2)
    st.pyplot(fig2)
    
    # =========================
    # 🔥 OVERALL PREDICTION
    # =========================
    st.subheader("🧠 Overall Prediction")

    full_spec = cv2.resize(spec, (config.IMG_WIDTH, config.IMG_HEIGHT))
    full_spec = (full_spec - full_spec.min()) / (full_spec.max() - full_spec.min())

    full_spec = np.expand_dims(full_spec, axis=-1)
    full_spec = np.expand_dims(full_spec, axis=0)

    full_pred = model.predict(full_spec)[0][0]

    if full_pred > 0.5:
        st.error(f"⚠️ Abnormal ({full_pred:.2f})")
    else:
        st.success(f"✅ Normal ({full_pred:.2f})")

    st.progress(float(full_pred))

    # =========================
# 🔍 SEGMENT-WISE ANALYSIS
# =========================

    st.subheader("🔍 Segment-wise Analysis")

    segments = segment_audio(y, config.SAMPLES)

    predictions = []
    segment_labels = []

    for i, seg in enumerate(segments):
        
        seg = fix_length(seg, config.SAMPLES)
        
        seg_spec = audio_to_spectrogram(seg)
        
        seg_spec = cv2.resize(seg_spec, (config.IMG_WIDTH, config.IMG_HEIGHT))
        seg_spec = (seg_spec - seg_spec.min()) / (seg_spec.max() - seg_spec.min())
        
        seg_spec = np.expand_dims(seg_spec, axis=-1)
        seg_spec = np.expand_dims(seg_spec, axis=0)
        
        pred = model.predict(seg_spec)[0][0]
        predictions.append(pred)
        
        label = "Abnormal" if pred > 0.5 else "Normal"
        segment_labels.append(label)
        
        # 🔥 Highlight abnormal
        if label == "Abnormal":
            st.error(f"🔴 Segment {i+1}: {label} ({pred:.2f})")
        else:
            st.success(f"🟢 Segment {i+1}: {label} ({pred:.2f})")


    # =========================
    # 📊 CONFIDENCE GRAPH
    # =========================

    st.subheader("📊 Segment Confidence Graph")

    df = pd.DataFrame({
        "Segment": [f"S{i+1}" for i in range(len(predictions))],
        "Confidence": predictions
    })

    st.bar_chart(df.set_index("Segment"))
        # =========================
    # 📊 FINAL DECISION
    # =========================
    st.subheader("📊 Final Decision (Segment-Based)")

    avg_pred = np.mean(predictions)

    if avg_pred > 0.5:
        st.error(f"⚠️ Abnormal ({avg_pred:.2f})")
    else:
        st.success(f"✅ Normal ({avg_pred:.2f})")

    st.progress(float(avg_pred))