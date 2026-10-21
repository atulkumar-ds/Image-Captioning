# 🖼️ Image Captioning - V1 (CNN + LSTM Baseline)

## 📌 Overview

Version 1 is a baseline image captioning system built using a **pretrained InceptionV3 CNN** as the image encoder and an **LSTM decoder** for caption generation.

The objective of this project is to implement and benchmark an end-to-end image captioning pipeline, including data preprocessing, feature extraction, model training, greedy caption generation, and BLEU score evaluation.

---

## 🚀 Features

- Image feature extraction using **InceptionV3** (global 2048-D feature vector)
- Caption preprocessing with preservation of single-letter words (`a`, `i`)
- Frequency-filtered vocabulary creation
- Custom JSON-serialized tokenizer
- Custom Keras Data Generator with batch padding
- CNN + LSTM encoder-decoder fusion architecture
- Caption generation using Greedy Search
- Leak-aware corpus BLEU score evaluation (BLEU-1 to BLEU-4)
- Training history visualization utility
- Interactive Streamlit demonstration app

---

# 🏗 Architecture

```
                Image (299 x 299 x 3)
                          │
                     InceptionV3
                          │
                2048-D Feature Vector
                          │
                    Dropout (0.5)
                          │
                     Dense (256)
                          │
                          ├──────────────┐
                          │              │
Caption → Embedding → Dropout (0.5) → LSTM (256)
                          │              │
                          └──────┬───────┘
                                 │
                               Add()
                          │
                     Dense (256)
                          │
                    Softmax Layer
                          │
                    Predicted Word
```

---

# 📂 Dataset & Splits

- **Dataset:** Flickr8k (~8,000 images, 5 captions per image).
- **Canonical Split (`splits.json`):**
  - **Train:** 5,663 images (70%)
  - **Validation:** 1,213 images (15%)
  - **Test:** 1,215 images (15%)
  - Image-level partition guaranteeing zero caption leakage across splits.
- **Historical Split:** 90% train / 10% validation split historically used for `best_caption_model.keras`.

---

# 🛠 Tech Stack

- Python
- TensorFlow / Keras (>= 2.10.0)
- NumPy
- Matplotlib
- NLTK (for corpus BLEU with brevity penalty and smoothing)
- Streamlit

---

# 📁 Project Structure

```text
V1_Baseline_CNN_LSTM/
├── app/
│   └── app.py                  # Streamlit web application
├── data/
│   ├── captions/captions.txt   # Raw Flickr8k captions
│   ├── images/                 # Image files
│   └── features/               # Extracted 2048-D feature vectors (.npz)
├── models/
│   ├── best_caption_model.keras # Historical trained baseline checkpoint
│   ├── tokenizer.json          # Historical vocabulary tokenizer (2,814 tokens)
│   └── history.json            # Historical training metrics log
├── plots/                      # Training loss and accuracy plots
├── src/
│   ├── bleu_score.py           # Corpus BLEU evaluation with split verification
│   ├── cnn_features.py         # Lazy InceptionV3 feature extraction
│   ├── dataset.py              # Raw caption loading and cleaning
│   ├── data_generator.py       # Keras sequence data generator
│   ├── evaluate.py             # Single-sample prediction evaluation
│   ├── generate_caption.py     # Greedy decoding and inference pipeline
│   ├── model.py                # Keras Functional API model definition
│   ├── plot_history.py         # Loss and accuracy curve generator
│   ├── preprocess.py           # Text normalization routines
│   ├── sequences.py            # Sequence padding and truncation
│   ├── tokenizer.py            # Tokenizer construction
│   ├── train.py                # Baseline training pipeline
│   └── vocabulary.py           # Vocabulary builder
├── requirements.txt            # Package dependencies
└── README.md
```

---

## 🛠️ Usage Instructions

### 1. Training, Reproducibility, and Dry-Run Verification

The training script supports deterministic seeding, experiment tracking, atomic history logging, and non-mutating dry-run verification:

- **Reproducibility Seeding (`--seed`, default: 42):** Sets seeds for Python `random`, `numpy.random`, and `tensorflow`. Controls layer initialization, dropout masks, and generator batch shuffling.
  > [!NOTE]
  > *Limits of Reproducibility:* Algorithmic seeds guarantee repeatable pseudorandom sequences across Python, NumPy, and TensorFlow. However, exact bit-for-bit numerical outputs across different GPUs, CUDA/cuDNN driver versions, or multithreaded CPU instruction schedules (e.g., parallel reductions in oneDNN) are subject to floating-point rounding nondeterminism. Split membership is strictly invariant to seed values and is read directly from `splits.json`. Validation evaluation is deterministic as `val_generator` sets `shuffle=False`.
- **Atomic History Persistence:** `history.json` is safely updated after every completed epoch using atomic temporary files (`.tmp` + `os.replace`), preventing corrupted JSON if execution is interrupted.
- **Synchronized Checkpoint Tracking:** The best checkpoint is chosen by minimizing `val_loss`. When a new minimum is achieved, its SHA-256 fingerprint, epoch number, and validation loss are atomically written into `config.json`.
- **Experiment Protection (`--allow_overwrite_experiment`):** Prevents accidental overwrites of existing completed experiments.

```bash
# Non-mutating validation-only dry-run (validates splits, vocabulary, and paths without training or modifying files):
python V1_Baseline_CNN_LSTM/src/train.py --dry_run

# Retrain V1 on the leak-free canonical split with deterministic seed:
python V1_Baseline_CNN_LSTM/src/train.py \
    --split canonical \
    --output_dir V1_Baseline_CNN_LSTM/experiments/v1_canonical \
    --seed 42 \
    --epochs 40 \
    --batch_size 64
```

### 2. Evaluating on Held-Out Splits (Corpus BLEU)
Evaluate on unseen images using corpus BLEU-1 through BLEU-4 via `bleu_score.py`:

> [!IMPORTANT]
> **Data Leakage Notice:** Historical checkpoint `best_caption_model.keras` was trained on `historical_v1_split["train"]`, which contains 405 images that are in `canonical_split["test"]`. Therefore, evaluating the historical model on `canonical_test` constitutes data leakage. Historical models must ONLY be evaluated on their matching historical validation split (`historical_v1_val`). Canonical test evaluation is reserved exclusively for newly trained canonical models.

```bash
# Evaluate historical checkpoint on historical validation split (Valid):
python V1_Baseline_CNN_LSTM/src/bleu_score.py --split historical_v1_val

# Evaluate a newly retrained canonical model on the canonical test split:
python V1_Baseline_CNN_LSTM/src/bleu_score.py \
    --split canonical_test \
    --trained_on canonical \
    --model V1_Baseline_CNN_LSTM/experiments/v1_canonical/models/best_caption_model.keras \
    --tokenizer V1_Baseline_CNN_LSTM/experiments/v1_canonical/models/tokenizer.json
```

### 3. Evaluating a Single Sample (Inference)
> [!NOTE]
> `evaluate.py` performs qualitative inference for a single image, contrasting with the corpus-level statistical benchmarking in `bleu_score.py`.

```bash
# Run single-image evaluation with working defaults (uses default models/best_caption_model.keras):
python V1_Baseline_CNN_LSTM/src/evaluate.py --image 1000268201_693b08cb0e.jpg

# Or explicitly specify model and tokenizer (e.g., from a retrained canonical experiment):
python V1_Baseline_CNN_LSTM/src/evaluate.py \
    --image 1000268201_693b08cb0e.jpg \
    --model V1_Baseline_CNN_LSTM/experiments/v1_canonical/models/best_caption_model.keras \
    --tokenizer V1_Baseline_CNN_LSTM/experiments/v1_canonical/models/tokenizer.json

# Or run the greedy caption inference test script directly:
python V1_Baseline_CNN_LSTM/src/generate_caption.py
```

### 4. Plotting Training Curves
```bash
python V1_Baseline_CNN_LSTM/src/plot_history.py \
    --history V1_Baseline_CNN_LSTM/models/history.json \
    --output_dir V1_Baseline_CNN_LSTM/plots
```

### 5. Running the Streamlit App
```bash
streamlit run V1_Baseline_CNN_LSTM/app/app.py
```

---

# 📊 Results (Historical Checkpoint)

| Metric | Historical Held-Out Score |
|:-------|:--------------------------|
| BLEU-1 | **0.4830** |
| BLEU-2 | **0.2961** |
| BLEU-3 | **0.1774** |
| BLEU-4 | **0.1012** |

---

# ⚖️ Fair Model Comparison Protocol

- **Target and Loss Semantics:** V1 generates prefix-target pairs where the supervised target is a single valid caption token ($w_i \in \{1, \dots, V-1\}$). Zero-padding tokens appear only in the input prefix and are masked via `Embedding(mask_zero=True)`. Supervised targets in V1 **never** include `<pad>`, so padding tokens do not pull down V1 loss. Both V1 and V2 optimize cross-entropy exclusively across valid, non-padding tokens.
- **Why Evaluation Requires Held-Out Corpus BLEU:** Cross-entropy measures single-step predictive probability under teacher forcing with ground-truth prefixes. At test time, captions are generated greedily without ground truth, where prediction errors compound. Therefore, model architecture quality must be benchmarked using Corpus BLEU (BLEU-1 through BLEU-4) on the mutually held-out `canonical_split["test"]` partition after training on `canonical_split["train"]`.

---

# ⚠️ Baseline Limitations

- **Global Context Bottleneck:** Compress the entire image into a single 2048-D vector, discarding fine-grained spatial information.
- **No Alignment:** Cannot focus on specific image objects or regions while producing specific words.
- **Greedy Decoding:** Sensitive to local early token errors without beam search recovery.
- *Addressed in V2 through Bahdanau spatial attention.*
