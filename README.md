# ❤️ Heart Sound Disease Detection

## 📌 Overview

This project detects abnormal heart sounds from audio recordings using deep learning.
It processes raw audio signals, converts them into spectrograms, and classifies them using a CNN model.

---

## 🎯 Objective

To build an AI system that can assist in identifying abnormal heart conditions (like murmurs, arrhythmias) from heart sound recordings.

---

## ⚙️ What We Did

* Collected heart sound dataset (audio + labels)
* Performed **audio segmentation (3-sec chunks)**
* Applied **data augmentation** (noise, pitch shift, time stretch)
* Handled **class imbalance**
* Converted audio → **Mel Spectrograms**
* Built and trained a **CNN model**
* Achieved ~**83% validation accuracy**
* Implemented:

  * ✅ Overall prediction
  * ✅ Segment-wise prediction
  * ✅ Confidence scoring
* Developed an **interactive Streamlit web app**

---

## 🧠 Model Pipeline

Audio → Segmentation → Spectrogram → CNN → Prediction

---

## 📊 Features

* Upload heart sound (.wav)
* View waveform and spectrogram
* Get:

  * Overall prediction (Normal / Abnormal)
  * Segment-wise analysis
  * Confidence visualization

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* Librosa
* OpenCV
* Scikit-learn
* Streamlit

---

## 🚀 How to Run

```bash
git clone https://github.com/your-username/heart-sound-detection.git
cd heart-sound-detection
pip install -r requirements.txt
streamlit run app.py
```

---

## ⚠️ Note

This project is for educational purposes only and not intended for medical diagnosis.

---

## 👨‍💻 Author

Vikas Pathade
B.Tech AI & DS
