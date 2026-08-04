# 🖼️ Image Captioning

A complete deep learning project that explores the evolution of image captioning models—from a simple CNN + LSTM baseline to modern attention-based and transformer-based architectures.

The goal of this repository is not only to build an image captioning system but also to document the complete learning journey by implementing increasingly advanced models.

---

## 🚀 Project Roadmap

| Version | Model | Status |
|---------|-------|--------|
| ✅ V1 | CNN + LSTM Baseline | Completed |
| 🚧 V2 | CNN + Bahdanau Attention + LSTM | In Progress |
| ⏳ V3 | Transformer Decoder | Planned |
| ⏳ V4 | Vision Transformer (ViT) + Transformer | Planned |
| ⏳ V5 | BLIP / Pretrained Vision-Language Model | Planned |
| ⏳ V6 | Video Captioning | Planned |
| ⏳ V7 | Real-Time Captioning | Planned |

---

# 📂 Repository Structure

```text
Image-Captioning/
│
├── V1_Baseline_CNN_LSTM/
│   ├── app/
│   ├── models/
│   ├── plots/
│   ├── src/
│   ├── README.md
│   └── requirements.txt
│
├── V2_Attention/
│   ├── model/
│   ├── src/
│   └── README.md
│
└── README.md
```

---

# 📖 Versions

## ✅ V1 — CNN + LSTM Baseline

### Architecture

```
Image
   │
InceptionV3
   │
Image Feature Vector
   │
LSTM Decoder
   │
Caption
```

### Features

- Pretrained InceptionV3 Encoder
- LSTM Decoder
- Teacher Forcing
- Vocabulary Building
- Caption Generation
- BLEU Score Evaluation
- Training History Visualization

---

## 🚧 V2 — Attention Mechanism

### Architecture

```
Image
   │
CNN
   │
Feature Maps
   │
Encoder
   │
Bahdanau Attention
   │
LSTM Decoder
   │
Caption
```

### Improvements over V1

- Spatial image features
- Bahdanau Attention
- Better alignment between image regions and generated words
- Improved caption quality
- Attention visualization (planned)

---

# 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- InceptionV3
- LSTM
- Bahdanau Attention
- BLEU Evaluation

---

# 📊 Dataset

**Flickr8k Dataset**

- Images
- Multiple captions per image
- Used for training and evaluation

---

# 📈 Future Improvements

- Beam Search Decoding
- Scheduled Sampling
- Transformer Decoder
- Vision Transformer (ViT)
- BLIP
- CLIP Integration
- Video Captioning
- Real-Time Captioning
- Web Application
- API Deployment

---

# 🎯 Learning Objectives

This repository demonstrates:

- Deep Learning
- Sequence Models
- Attention Mechanisms
- Transformer Architectures
- Model Evaluation
- Transfer Learning
- End-to-End AI Project Development

---

# 📷 Sample Workflow

```
Image
   │
CNN Feature Extraction
   │
Encoder
   │
Attention
   │
Decoder
   │
Generated Caption
```

---

# 📌 Current Progress

- ✅ Baseline CNN + LSTM completed
- 🚧 Attention mechanism under development
- ⏳ Transformer implementation planned
- ⏳ Vision-language models planned

---

# 👨‍💻 Author

**Atul Kumar**

B.Tech – Artificial Intelligence & Data Science  
IIITDM Kurnool

- GitHub: https://github.com/atulkumar-ds
- LinkedIn: linkedin.com/in/atul-kumar-iiitdm/


## ⭐ If you find this project helpful, consider giving it a star!