import os
import numpy as np
import pandas as pd
import librosa
import random
from tqdm import tqdm
import sys
# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config

# =========================
# LOAD DATA
# =========================
def load_data():
    df = pd.read_csv(config.CSV_PATH)
    
    # Create filename column
    df['filename'] = df['Heart Sound ID'] + ".wav"
    
    # Binary labels
    df['label'] = df['Heart Sound Type'].apply(
        lambda x: 0 if x == 'Normal' else 1
    )
    
    return df


# =========================
# SEGMENT AUDIO
# =========================
def segment_audio(file_path):
    y, sr = librosa.load(file_path, sr=config.SAMPLE_RATE)
    
    segment_length = config.SAMPLES  # e.g. 3 sec
    segments = []
    
    for start in range(0, len(y), segment_length):
        end = start + segment_length
        
        segment = y[start:end]
        
        # Skip too short segments
        if len(segment) < segment_length:
            continue
        
        segments.append(segment)
    
    return segments

def augment_audio(audio):
    augmented = []
    
    noise = audio + 0.005 * np.random.randn(len(audio))
    augmented.append(noise)
    
    pitch = librosa.effects.pitch_shift(audio, sr=config.SAMPLE_RATE, n_steps=2)
    augmented.append(pitch)
    
    stretch = librosa.effects.time_stretch(audio, rate=0.8)
    augmented.append(stretch)
    
    return augmented

def fix_length(audio, target_length):
    if len(audio) > target_length:
        return audio[:target_length]
    elif len(audio) < target_length:
        return np.pad(audio, (0, target_length - len(audio)))
    return audio
# =========================
# CREATE SEGMENT DATASET
# =========================
def create_segments(df):
    X = []
    y = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        file_path = os.path.join(config.DATA_PATH, row['filename'])
        
        if not os.path.exists(file_path):
            continue
        
        segments = segment_audio(file_path)
        
        for seg in segments:
        # ✅ FIX original segment
            seg = fix_length(seg, config.SAMPLES)
            
            X.append(seg)
            y.append(row['label'])
            
            # 🔥 Augment ONLY normal class
            if row['label'] == 0:
                aug_segments = augment_audio(seg)
                
                for aug in aug_segments:
                    # ✅ FIX augmented segment
                    aug = fix_length(aug, config.SAMPLES)
                    
                    X.append(aug)
                    y.append(0)  # Augmented samples are also normal
    
    return np.array(X, dtype=np.float32), np.array(y)


# =========================
# BALANCE DATASET
# =========================
def balance_data(X, y):
    # Separate classes
    normal_idx = np.where(y == 0)[0]
    abnormal_idx = np.where(y == 1)[0]
    
    # Downsample majority class
    min_count = min(len(normal_idx), len(abnormal_idx))
    
    normal_sample = np.random.choice(normal_idx, min_count, replace=False)
    abnormal_sample = np.random.choice(abnormal_idx, min_count, replace=False)
    
    final_idx = np.concatenate([normal_sample, abnormal_sample])
    
    np.random.shuffle(final_idx)
    
    return X[final_idx], y[final_idx]


# =========================
# MAIN PIPELINE
# =========================
def run_preprocessing():
    print("Loading data...")
    df = load_data()
    
    print("Creating segments...")
    X, y = create_segments(df)
    
    print("Before balancing:", np.bincount(y))
    
    print("Balancing dataset...")
    X_bal, y_bal = balance_data(X, y)
    
    print("After balancing:", np.bincount(y_bal))
    
    return X_bal, y_bal


if __name__ == "__main__":
    X, y = run_preprocessing()
    print("Final dataset shape:", X.shape)