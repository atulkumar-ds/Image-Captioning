# 🖼️ Image Captioning - V1 (CNN + LSTM Baseline)

## 📌 Overview

Version 1 is a baseline image captioning system built using a **pretrained InceptionV3 CNN** as the image encoder and an **LSTM decoder** for caption generation.

The objective of this project is to learn and implement the complete image captioning pipeline from scratch, including data preprocessing, feature extraction, model training, caption generation, and evaluation.

---

## 🚀 Features

- Image feature extraction using **InceptionV3**
- Caption preprocessing and cleaning
- Vocabulary creation
- Custom tokenizer
- Training sample generation
- Custom Keras Data Generator
- CNN + LSTM encoder-decoder architecture
- Caption generation using Greedy Search
- BLEU Score evaluation
- Training history visualization
- Modular project structure

---

# 🏗 Architecture

```
                Image
                  │
          InceptionV3
                  │
      2048-D Feature Vector
                  │
               Dense
                  │
                  ├──────────────┐
                  │              │
Caption → Embedding → LSTM ──────┘
                  │
                Add
                  │
               Dense
                  │
             Softmax Layer
                  │
            Predicted Word
```

---

# 📂 Dataset

**Dataset:** Flickr8k

- ~8,000 images
- 5 captions per image
- Total captions: ~40,000

Each image is paired with multiple human-written captions.

---

# 🛠 Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- NLTK
- InceptionV3 (ImageNet pretrained)

---

# 📁 Project Structure

```
V1_Baseline_CNN_LSTM
│
├── app/
├── data/
│   ├── captions/
│   ├── images/
│   └── features/
│
├── models/
│
├── notebooks/
│
├── plots/
│
├── src/
│   ├── cnn_features.py
│   ├── dataset.py
│   ├── vocabulary.py
│   ├── tokenizer.py
│   ├── sequences.py
│   ├── data_generator.py
│   ├── model.py
│   ├── train.py
│   ├── generate_caption.py
│   ├── evaluate.py
│   ├── bleu_score.py
│   └── plot_history.py
│
├── README.md
└── requirements.txt
```

---

# ⚙ Pipeline

## 1. Data Preprocessing

- Load Flickr8k captions
- Clean text
- Remove punctuation
- Convert to lowercase
- Add `<start>` and `<end>` tokens

---

## 2. Vocabulary Creation

- Count word frequencies
- Remove rare words
- Create:
  - `word_to_index`
  - `index_to_word`

---

## 3. Image Feature Extraction

Encoder:

**InceptionV3**

- ImageNet pretrained
- Top layer removed
- Global Average Pooling
- Output Feature Size:

```
2048
```

Extracted features are stored as:

```
image_features.npz
```

to avoid repeated CNN computation.

---

## 4. Training Sample Generation

Each caption is converted into multiple training examples.

Example:

Caption:

```
<start> a dog is running <end>
```

Training pairs:

```
<start>                → a

<start> a              → dog

<start> a dog          → is

<start> a dog is       → running

<start> a dog is running → <end>
```

---

## 5. Model

### Image Branch

- Input (2048)
- Dropout
- Dense (256)

---

### Caption Branch

- Embedding
- Dropout
- LSTM (256)

---

### Fusion

Image features and caption representation are combined using:

```
Add()
```

---

### Output

Dense layer with Softmax activation predicts the next word.

---

# 📈 Training

Optimizer

- Adam

Loss Function

- Sparse Categorical Crossentropy

Callbacks

- ModelCheckpoint
- EarlyStopping
- ReduceLROnPlateau

---

# 📊 Results

## Training Accuracy

≈ **38%**

Validation Accuracy

≈ **34%**

---

## BLEU Scores

| Metric | Score |
|---------|------:|
| BLEU-1 | **0.4830** |
| BLEU-2 | **0.2961** |
| BLEU-3 | **0.1774** |
| BLEU-4 | **0.1012** |

---

# 📷 Sample Prediction

Ground Truth

```
<start> man dressed in batman costume <end>
```

Predicted

```
man in blue shirt and white hat is sitting on the sidewalk
```

This demonstrates that the baseline model learns general sentence structures but often misses fine-grained visual details.

---

# ⚠ Limitations

- Uses a single global image feature vector.
- Cannot focus on different regions of an image.
- Greedy decoding may produce suboptimal captions.
- BLEU-4 score remains relatively low.

---

# 🚀 Future Improvements (Version 2)

The next version will introduce:

- Bahdanau Attention
- Spatial image features
- Better caption quality
- Improved BLEU scores

Repository Progress:

```
✅ V1 - CNN + LSTM Baseline
🚧 V2 - Attention Mechanism
⏳ V3 - Beam Search
⏳ V4 - Transformer
⏳ V5 - BLIP
```

---

# 👨‍💻 Author

**Atul Kumar**

B.Tech – Artificial Intelligence & Data Science

IIITDM Kurnool

GitHub: https://github.com/atulkumar-ds