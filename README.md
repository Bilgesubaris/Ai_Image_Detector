# Explainable Deepfake Detection System (AI Image Detector)

This repository contains an explainable deep-learning-based system designed to detect deepfake and AI-generated images using **Vision Transformers (ViT)** and **Error Level Analysis (ELA)**. 

The project focuses on both high-accuracy detection and explainability, allowing users to understand *why* an image is classified as real or fake by visualizing the digital artifacts and model attention maps.
## 🧠 Methodology

### 1. Error Level Analysis (ELA)
ELA works by resaving an image at a specific compression rate (e.g., 95%) and calculating the absolute difference between the original and the resaved image. AI-generated or edited sections typically display higher error levels due to inconsistent compression rates, exposing local manipulation artifacts.

### 2. Domain-Specific Feature Extraction vs. Deep Learning
Unlike generalized Large Language Models (LLMs) or abstract semantic feature extractors, this system treats feature extraction as a **domain-specific task**. 

- **Classic/Statistical Forensics:** Traditional localized artifact detection methods (like ELA) provide a mathematically stable and deterministic foundation by isolating pixel-level compression noise and quantization table discrepancies.
- **Vision Transformer (ViT) Architecture:** Instead of relying on standard CNNs, the ELA-preprocessed image is divided into fixed-size patches, flattened, and processed through self-attention mechanisms. This allows the network to learn global context and map long-range dependencies across these localized compression anomalies, leading to robust and explainable deepfake classification.

  
## 🚀 Key Features

- **Hybrid Detection Framework:** Combines classical digital forensics (Error Level Analysis) with state-of-the-art Deep Learning (Vision Transformers).
- **Error Level Analysis (ELA):** Preprocesses images to highlight compression anomalies, making pixel-level manipulations visible.
- **Vision Transformer (ViT) Core:** Leverages self-attention mechanisms to capture global context and long-range dependencies across image patches.
- **Explainable AI (XAI):** Visualizes the model's decision-making process through attention maps overlaid on ELA artifacts.
- **Modular Architecture:** Clean separation of data preprocessing, model training, and inference application.

## 📁 Project Structure

```text
├── app.py                # Streamlit or Gradio web application for inference
├── train.py              # Training script for the Vision Transformer model
├── ela.py                # Error Level Analysis (ELA) preprocessing utilities
├── models/               # Directory containing saved model weights (.pth)
├── utils/                # Helper functions for data loading and metrics
└── README.md             # Project documentation

