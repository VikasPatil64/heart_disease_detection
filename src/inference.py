import os
import sys

# Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import librosa
import cv2
import tensorflow as tf

import config


# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model(config.MODEL_PATH + "heart_model.h5")


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
# AUDIO → SPECTROGRAM
# =========================
def audio_to_spectrogram(audio):
    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=config.SAMPLE_RATE,
        n_mels=config.N_MELS
    )
    
    spec_db = librosa.power_to_db(spec, ref=np.max)
    
    spec_resized = cv2.resize(
        spec_db,
        (config.IMG_WIDTH, config.IMG_HEIGHT)
    )
    
    spec_norm = (spec_resized - spec_resized.min()) / (spec_resized.max() - spec_resized.min())
    
    spec_norm = np.expand_dims(spec_norm, axis=-1)
    
    return spec_norm


# =========================
# PREDICT FUNCTION
# =========================
def predict_audio(file_path):
    y, sr = librosa.load(file_path, sr=config.SAMPLE_RATE)
    
    # Take first 3 sec
    y = fix_length(y, config.SAMPLES)
    
    spec = audio_to_spectrogram(y)
    
    spec = np.expand_dims(spec, axis=0)
    
    pred = model.predict(spec)[0][0]
    
    if pred > 0.5:
        return "Abnormal", pred
    else:
        return "Normal", pred


# =========================
# TEST
# =========================
if __name__ == "__main__":
    
    test_file = input("Enter audio file path: ")
    
    label, confidence = predict_audio(test_file)
    
    print(f"\nPrediction: {label}")
    print(f"Confidence: {confidence:.4f}")