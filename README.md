# 🖼️ Image Captioning

A modular, reproducible deep learning project investigating the progression of image captioning architectures on the Flickr8k dataset—from a CNN + LSTM baseline to spatial attention mechanisms and beyond.

---

## 🚀 Project Roadmap

| Version | Model Architecture | Status | Highlights |
|:-------:|:-------------------|:------:|:-----------|
| **V1**  | InceptionV3 (Global) + LSTM | ✅ Completed | 2048-D global feature vector, Keras Functional API, greedy decoding |
| **V2**  | InceptionV3 (Spatial) + Bahdanau Attention + LSTM | ✅ Prepared & Audited | $8 \times 8$ spatial grids, additive attention, valid-token normalized loss |
| **V3**  | Transformer Decoder (Self-Attention & Cross-Attention) | ⏳ Planned | Multi-head attention, positional encodings |
| **V4**  | Vision Transformer (ViT) Encoder + Transformer Decoder | ⏳ Planned | Patch-based vision encoding without CNN bottleneck |
| **V5**  | Multimodal Pretrained Models (BLIP / CLIP) | ⏳ Planned | Contrastive vision-language pretraining and zero-shot alignment |

---

## 📂 Repository Structure

```text
Image-Captioning/
├── comparison_app.py           # CaptionLab unified side-by-side comparison web app
├── requirements.txt            # Unified project dependency specification
├── splits.json                 # Canonical 70/15/15 & historical split manifests
├── README.md                   # Project overview and quickstart guide
│
├── tests/                      # Automated unit and regression test suite
│   ├── test_evaluation.py      # Splits disjointness, leakage detection & BLEU tests
│   ├── test_training_semantics.py # Text preprocessing & masked loss normalization tests
│   └── test_code_correctness.py   # Padding/truncation, attention shapes & lazy CNN tests
│
├── V1_Baseline_CNN_LSTM/       # Baseline model version
│   ├── app/                    # Streamlit web application
│   ├── data/                   # Flickr8k captions, images & extracted features
│   ├── models/                 # Historical weights (.keras) and tokenizer
│   ├── plots/                  # Loss and accuracy curves
│   ├── src/                    # Source modules (features, dataset, model, train, bleu)
│   ├── requirements.txt
│   └── README.md
│
└── V2_Attention/               # Attention-based model version
    ├── app.py                  # Streamlit web application
    ├── models/                 # Historical weights (.weights.h5) and tokenizer
    ├── experiments/            # Destination for new canonical training runs
    ├── plots/                  # Masked loss curves
    ├── src/                    # Source modules (attention, decoder, model, loss, train)
    ├── requirements.txt
    └── README.md
```

---

## 📊 Dataset & Canonical Splits

The project uses the **Flickr8k** benchmark (8,091 images, 5 reference captions per image).

To prevent evaluation data leakage, all images and their associated captions are partitioned into a deterministic, mutually disjoint canonical split in `splits.json`:
- **Training Set:** 5,663 images (70%)
- **Validation Set:** 1,213 images (15%)
- **Test Set:** 1,215 images (15%)

> [!IMPORTANT]
> **Evaluation Split Compatibility & Leak-Free Benchmarking:**
> - Historical checkpoints (`models/best_caption_model.keras` and `models/v2_attention.weights.h5`) were trained on historical split partitions.
> - **Critical Finding:** Canonical test images heavily overlap historical training sets (405 canonical test images were in V1 train; 1,215 in V2 train). Therefore, evaluating historical checkpoints on `canonical_test` constitutes severe data contamination.
> - **Enforced Rule:** Historical checkpoints must ONLY be evaluated on their matching historical validation splits (`historical_v1_val` and `historical_v2_val`).
> - The canonical test split (`canonical_test`) is reserved strictly for newly trained models produced by canonical experiments.

---

## 🔬 Model Comparison & Fair Evaluation Protocol

| Feature | V1 Baseline | V2 Attention |
|:---|:---|:---|
| **Visual Encoder** | InceptionV3 (Global 2048-D) | InceptionV3 `mixed10` ($8 \times 8 \times 2048 \rightarrow 64 \times 2048$) |
| **Attention Mechanism** | None | Bahdanau (Additive) Attention |
| **Decoder** | LSTM (256 units) | Attention-augmented LSTM (256 units) |
| **Loss Formulation** | Sparse Categorical Cross-Entropy (per prefix-target pair) | Masked Sparse Categorical Cross-Entropy (per sequence, normalized by valid tokens) |
| **Supervised Target Semantics** | Single valid next token per prefix; `<pad>` is **never** a target | Full sequence with trailing `<pad>` explicitly masked out ($m_{i,t} = 0$) |
| **Feature Extraction** | Pre-extracted `.npz` | Streaming / `.npy` with lazy loading |
| **Historical BLEU-1** | 0.4830 | 0.4617 |
| **Historical BLEU-4** | 0.1012 | 0.0881 |

> [!NOTE]
> **Loss Semantics & Cross-Architecture Evaluation:**
> - **Both models train exclusively on valid tokens:** In V1, each sample is an autoregressive prefix-target pair predicting an actual caption token ($w_i \in \{1, \dots, V-1\}$); padding tokens appear solely as input padding masked by `Embedding(mask_zero=True)` and are **never** supervised targets. In V2, captions are processed as sequence tensors where trailing `<pad>` tokens are dynamically masked out and normalized by the count of valid non-padding tokens.
> - **Loss Aggregation Differences:** V1 aggregates loss at the sample level (each sample is one valid token), yielding the exact dataset-level mean cross-entropy per valid token. V2 normalizes loss per-batch by valid tokens, then averages batch losses across the epoch.
> - **Why Evaluation Requires Held-Out Corpus BLEU:** Training and validation loss measure one-step-ahead cross-entropy under teacher forcing (where the true ground truth prefix is always supplied). During actual inference, captions are generated autoregressively without ground truth, where prediction errors compound. Therefore, model performance cannot be judged by training loss curves alone; rigorous comparison must be conducted using Corpus BLEU (BLEU-1 through BLEU-4) on the identical held-out test split (`canonical_split["test"]`) against identically preprocessed reference captions.

---

## 🔒 Reproducibility & Experiment Tracking

The training pipelines for both V1 and V2 incorporate auditable experiment tracking and reproducibility controls:

1. **Seed Control (`--seed`, default: 42):**
   - Configures deterministic random seeds across Python `random`, NumPy `np.random`, and TensorFlow `tf.random` / `tf.keras.utils.set_random_seed`.
   - Controls layer weight initialization, dropout masks, and sequence shuffling order in training data pipelines.
   - *Split Membership Invariance:* Dataset partition membership is strictly governed by `splits.json` and is completely invariant to random seeds.
   - *Deterministic Validation:* Validation data in both V1 and V2 uses `shuffle=False` for deterministic validation loss evaluation.
   - *Limits of Numerical Reproducibility:* Setting seeds guarantees repeatable pseudorandom sequences. However, exact bit-for-bit numerical reproducibility of floating-point weights and loss values across different GPU architectures, CUDA/cuDNN driver versions, or multithreaded CPU instruction schedules (e.g., parallel reductions in oneDNN) is subject to floating-point nondeterminism. Enforcing strict operational determinism via `tf.config.experimental.enable_op_determinism()` can incur significant performance overhead and may fail if an op lacks a deterministic kernel.
2. **Atomic Training History Persistence (`history.json`):**
   - Written atomically after every completed epoch via temporary swap files (`.tmp` + `os.replace`).
   - Protects against process termination or system power interruptions, ensuring the record contains clean, uncorrupted history up to the last finished epoch.
3. **Synchronized Checkpoint Tracking (`config.json`):**
   - Checkpoint selection rule: strictly tracks the lowest validation loss (`min(val_loss)`).
   - Whenever a new best checkpoint is saved, its SHA-256 `checkpoint_fingerprint`, current `completed_epochs`, and `best_val_loss` are atomically updated in `config.json`.
4. **Experiment Overwrite Protection (`--allow_overwrite_experiment`):**
   - Retraining aborts with an error if the destination directory contains an existing completed experiment, unless `--allow_overwrite_experiment` is explicitly supplied.

---

## 🛠️ Quickstart

### 1. Environment Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

> [!NOTE]
> **Dependency & Platform Compatibility:**
> - The verified active environment uses **TensorFlow 2.21.0** and **Keras 3.14.1**.
> - **Keras 3 Compatibility:** Standalone `keras>=3.0.0` requires `tensorflow>=2.16.0`. Earlier TensorFlow versions (< 2.16) bundled Keras 2.x and are incompatible with Keras 3.
> - **Windows vs. WSL2 GPU Support:** Native Windows builds of TensorFlow $\ge 2.11$ operate on CPU only. GPU-accelerated training requires **WSL2** (Ubuntu) with NVIDIA Container Toolkit / CUDA drivers.

### 2. Run Automated Verification Tests
Execute the lightweight unit test suite (covering preprocessing, masked loss normalization, splits disjointness, lazy loading, and model architectures without loading checkpoints or GPU):
```bash
python -m unittest discover tests
```

### 3. Controlled Model Training & Retraining
Both models can be trained reproducibly on the leak-free canonical dataset split with deterministic seeding:

```bash
# V1 Baseline training on canonical split:
python V1_Baseline_CNN_LSTM/src/train.py \
    --split canonical \
    --output_dir V1_Baseline_CNN_LSTM/experiments/v1_canonical \
    --seed 42 \
    --epochs 40 \
    --batch_size 64

# V2 Attention training on canonical split:
python V2_Attention/src/train.py \
    --split canonical \
    --output_dir V2_Attention/experiments/v2_canonical \
    --features_file V2_Attention/model/v2_features.npy \
    --seed 42 \
    --epochs 40 \
    --batch_size 32
```
*(To safely validate vocabulary, split manifests, and configurations without training or writing files, use `--dry_run`. If retraining into an existing experiment directory, supply `--allow_overwrite_experiment`.)*

### 4. Evaluate Models on Held-Out Splits (Corpus BLEU)
Corpus-level evaluation computes benchmark BLEU-1 through BLEU-4 scores across complete dataset partitions.

#### A. Historical Baseline Evaluation (Valid on Historical Splits Only)
```bash
# Evaluate V1 baseline on its historical validation split:
python V1_Baseline_CNN_LSTM/src/bleu_score.py --split historical_v1_val

# Evaluate V2 attention on its historical validation split:
python V2_Attention/src/evaluate.py --split historical_v2_val
```

#### B. Canonical Experiment Evaluation (For Models Retrained on Canonical Split)
```bash
# Evaluate newly trained canonical V1 model on canonical test split:
python V1_Baseline_CNN_LSTM/src/bleu_score.py \
    --split canonical_test \
    --trained_on canonical \
    --model V1_Baseline_CNN_LSTM/experiments/v1_canonical/models/best_caption_model.keras \
    --tokenizer V1_Baseline_CNN_LSTM/experiments/v1_canonical/models/tokenizer.json

# Evaluate newly trained canonical V2 model on canonical test split:
python V2_Attention/src/evaluate.py \
    --split canonical_test \
    --trained_on canonical \
    --weights V2_Attention/experiments/v2_canonical/models/v2_attention.weights.h5 \
    --tokenizer V2_Attention/experiments/v2_canonical/models/tokenizer.json
```

### 5. Single-Image Sample Inference Evaluation
Qualitative evaluation on individual image files is distinct from corpus-level BLEU scoring:
```bash
# V1 single-sample evaluation (uses default historical model and tokenizer; --model and --tokenizer are optional):
python V1_Baseline_CNN_LSTM/src/evaluate.py --image 416650559_cd08d3cd96.jpg

# V1 greedy caption inference test script:
python V1_Baseline_CNN_LSTM/src/generate_caption.py

# V2 attention caption inference test script:
python V2_Attention/src/generate_caption.py
```

### 6. Interactive Web Applications

#### A. CaptionLab — Unified Side-by-Side Model Comparison (Recommended)
Launch the unified comparison web application to evaluate both V1 Baseline and V2 Attention models simultaneously on any uploaded image:
```powershell
python -m streamlit run comparison_app.py
```
*Features: Dual-model real-time inference, monotonic latency benchmarks, token alignment & vocabulary similarity analysis, copy-to-clipboard, and JSON/TXT report exports.*

#### B. Standalone Version 1 Web Application (Baseline CNN + LSTM)
```powershell
streamlit run V1_Baseline_CNN_LSTM/app/app.py --server.port 8502
```

#### C. Standalone Version 2 Web Application (Spatial Attention)
```powershell
streamlit run V2_Attention/app.py --server.port 8503
```

---

## 👨‍💻 Author

**Atul Kumar**  
B.Tech – Artificial Intelligence & Data Science  
Indian Institute of Information Technology, Design and Manufacturing (IIITDM), Kurnool  
- GitHub: [atulkumar-ds](https://github.com/atulkumar-ds)  
- LinkedIn: [atul-kumar-iiitdm](https://linkedin.com/in/atul-kumar-iiitdm/)
