# 🧠 AI Image / Deepfake Detector

This project is a **machine learning-based image classification system** that detects whether an image is **REAL or FAKE (deepfake/forged)** using **Vision Transformer (ViT) + classical ML classifier**.

---

## 🚀 Features

- 🧠 Feature extraction using **Hugging Face ViT (google/vit-base-patch16-224)**
- 👤 Face detection using **MTCNN (face-only analysis)**
- 📊 Logistic Regression classifier
- 🌐 Web UI built with **Streamlit**
- ⚡ Real-time image prediction
- 🔍 Probability-based output (REAL / FAKE confidence)

---

## 🏗️ Architecture
Image → Face Detection → ViT Feature Extraction → Classifier → Prediction


---

## 📦 Installation

```bash
git clone https://github.com/Bilgesubaris/ai-image-detector.git
cd ai-image-detector

pip install -r requirements.txt
