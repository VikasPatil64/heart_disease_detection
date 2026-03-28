import os
import sys

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import librosa
import cv2
from tqdm import tqdm

import config


# =========================
# CONVERT AUDIO → SPECTROGRAM
# =========================
def audio_to_spectrogram(audio):
    # Generate Mel spectrogram
    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=config.SAMPLE_RATE,
        n_mels=config.N_MELS
    )
    
    # Convert to dB scale
    spec_db = librosa.power_to_db(spec, ref=np.max)
    
    return spec_db


# =========================
# PROCESS ALL SEGMENTS
# =========================
def create_spectrogram_dataset(X):
    images = []
    
    for audio in tqdm(X):
        spec = audio_to_spectrogram(audio)
        
        # Resize to fixed size (CNN input)
        spec_resized = cv2.resize(
            spec,
            (config.IMG_WIDTH, config.IMG_HEIGHT)
        )
        
        # Normalize (0–1)
        spec_norm = (spec_resized - spec_resized.min()) / (spec_resized.max() - spec_resized.min())
        
        # Add channel dimension
        spec_norm = np.expand_dims(spec_norm, axis=-1)
        
        images.append(spec_norm)
    
    return np.array(images)


# =========================
# MAIN PIPELINE
# =========================
def run_feature_extraction(X, y):
    print("Converting to spectrograms...")
    
    X_img = create_spectrogram_dataset(X)
    
    print("Final shape:", X_img.shape)
    
    return X_img, y


if __name__ == "__main__":
    # Dummy test (optional)
    print("Run this from training pipeline")