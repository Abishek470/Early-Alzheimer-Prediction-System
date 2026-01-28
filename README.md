# 🧠 Early Alzheimer's Prediction System

![Python](https://img.shields.io/badge/Python-3.10-blue) ![Framework](https://img.shields.io/badge/FastAPI-0.95-green) ![AI](https://img.shields.io/badge/TensorFlow-2.12-orange) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

An AI-powered non-invasive screening tool that detects early signs of Alzheimer's Disease from voice recordings. The system analyzes speech patterns—focusing on pauses, hesitation, and vocal tremors (jitter/shimmer)—using a hybrid Deep Learning ensemble.

## 🚀 Key Features

* **Dual-Model Architecture:** Combines **CNN-LSTM** (for spectral features) and **GRU-Attention** (for temporal hesitation patterns).
* **Ensemble Logic:** Averages predictions from both models to achieve robust accuracy (**F1-Score: 96%**).
* **Medical Biomarkers:** Explicitly extracts clinical features like **Jitter** (pitch instability), **Shimmer** (loudness fluctuation), and **Pause Duration**.
* **Robust Preprocessing:** Uses Voice Activity Detection (VAD) and High-Pass filtering to handle real-world noise.
* **Full-Stack Application:** Includes a **FastAPI** backend and a **React + Vite** frontend for easy user interaction.

## 🛠️ System Architecture

The system follows a 3-layer architecture:
1.  **Frontend (React):** User interface for audio recording/uploading and visualizing results.
2.  **Backend (FastAPI):** Handles authentication, file processing, and API routing.
3.  **AI Core (TensorFlow):** Preprocesses audio, extracts MFCCs + Biomarkers, and runs inference.

## 📂 Project Structure

### 1. The Core AI (`/`)
* `audio_preprocessing.py`: Cleans raw audio (denoising, VAD) and extracts Jitter/Shimmer.
* `data_pipeline.py`: Handles data augmentation (noise injection, pitch shifting) and prepares the dataset (300 time-steps, 47 features).
* `model.py`: Defines the **CNN-LSTM** architecture (Convolutional layers for feature extraction + LSTM for sequence memory).
* `model_gru.py`: Defines the **GRU-Attention** architecture (Focuses on specific hesitation frames).
* `train_model.py`: The training loop with Class Weighting, Early Stopping, and Learning Rate Reduction.
* `find_threshold.py`: Automatically calculates the optimal decision threshold (e.g., 0.54) to maximize the F1-Score.
* `evaluate_models_multi.py`: Generates ROC Curves, Confusion Matrices, and performance reports.

### 2. The Backend (`/`)
* `app.py`: The main **FastAPI** server. Exposes `/predict` and `/generate-report` endpoints.
* `auth.py`: Handles user registration and JWT-based login security.
* `inference.py`: The inference engine that loads trained models and runs predictions on new files.

## ⚙️ Installation & Setup

### Prerequisites
* Python 3.9+
* Node.js (for Frontend)

### 1. Clone the Repository
```bash
git clone [https://github.com/Abishek470/Early-Alzheimer-Prediction-System.git](https://github.com/Abishek470/Early-Alzheimer-Prediction-System.git)
cd Early-Alzheimer-Prediction-System
