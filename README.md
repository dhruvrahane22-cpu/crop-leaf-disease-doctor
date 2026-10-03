# 🌾 Crop Leaf-Disease Doctor: Image Classifier & Advisory Chatbot

An end-to-end computer vision and rule-based conversational AI system that accurately classifies plant foliage conditions across **38 distinct categories** and provides immediate, targeted agricultural treatment and prevention advice.

---

## 📌 Key Features

- **Multi-Model Image Classification:** Evaluates Custom CNNs, ResNet-50 fine-tuning paradigms, EfficientNet-B0, and Vision Transformers (ViT-B/16).
- **From-Scratch Mathematical Implementation:** Pure NumPy implementation of 2D Discrete Convolution and Max-Pooling layers without deep learning framework dependencies.
- **Model Interpretability (Grad-CAM):** Spatial activation visual heatmaps confirming model focus on leaf lesions rather than background artifacts.
- **Domain-Shift Robustness:** Enhanced field reliability against phone-captured field photos via heavy color jittering, random cropping, and Gaussian blur augmentations.
- **Rule-Based Advisory Chatbot:** Intent-routed retrieval engine delivering specific treatment and prevention advice based on official agricultural extension guidelines.
- **Decoupled System Architecture:** Asynchronous FastAPI backend for inference paired with an interactive Streamlit UI.

---

## 📊 Performance Benchmarks

Evaluated on the 15% holdout test split (54,303 total images across 38 classes):

- **Custom Leaf CNN (Baseline):** Test Acc = 89.4% | Macro-F1 = 0.882 | Params = 4.2M | Train Time = 18 mins | CPU Latency = 12 ms
- **ResNet-50 (Frozen):** Test Acc = 92.1% | Macro-F1 = 0.915 | Params = 23.5M | Train Time = 12 mins | CPU Latency = 45 ms
- **ResNet-50 (Fine-tune Last):** Test Acc = 96.3% | Macro-F1 = 0.960 | Params = 23.5M | Train Time = 22 mins | CPU Latency = 45 ms
- **ResNet-50 (Full Fine-tune):** Test Acc = 98.2% | Macro-F1 = 0.981 | Params = 23.5M | Train Time = 45 mins | CPU Latency = 45 ms
- **EfficientNet-B0:** Test Acc = 98.7% | Macro-F1 = 0.986 | Params = 5.3M | Train Time = 30 mins | CPU Latency = 22 ms
- **ViT-B/16 (Best Model):** Test Acc = 99.1% | Macro-F1 = 0.990 | Params = 86.6M | Train Time = 65 mins | CPU Latency = 110 ms

---

## 📁 Repository Structure

- **`app/api.py`**: FastAPI backend REST API server
- **`app/ui.py`**: Streamlit interactive frontend
- **`app/retriever.py`**: Rule-based knowledge base search engine
- **`data/`**: Dataset processing scripts and splits
- **`models/`**: Model architecture definitions & NumPy layers
- **`weights/`**: Saved model checkpoints (.pth)
- **`notebooks/`**: EDA, Training, and Grad-CAM notebooks
- **`requirements.txt`**: Project dependencies
- **`README.md`**: Project documentation

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/crop-disease-doctor.git](https://github.com/your-username/crop-disease-doctor.git)
cd crop-disease-doctor
