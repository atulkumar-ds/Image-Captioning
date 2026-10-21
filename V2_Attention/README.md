# 🖼️ V2 Attention-Based Image Captioning

An end-to-end Deep Learning image captioning system using **InceptionV3 Spatial Features + Bahdanau (Additive) Attention + LSTM Decoder**.

Version 2 upgrades the baseline V1 architecture by replacing the single global 2048-D feature vector with spatial feature maps, enabling the decoder to dynamically attend to relevant image regions at each step of caption generation.

---

## 🚀 Architecture Overview

```
                      Image (299 x 299 x 3)
                                │
                           InceptionV3 (mixed10 layer)
                                │
                    Spatial Features (8 x 8 x 2048)
                                │
                    Reshaped to (64 x 2048)
                                │
                    Dense Feature Projection (64 x 256)
                                │
                                ├───► [Bahdanau Attention] ◄─── Hidden State (h_{t-1})
                                │              │
                                │       Context Vector c_t (256-D)
                                │              │
Previous Token y_{t-1} ──► Embedding ─────────┴──► Concat [c_t, e_{t-1}]
                                                       │
                                                    LSTM Cell
                                                       │
                                                 Dense Projection
                                                       │
                                                 Vocabulary Logits
                                                       │
                                                 Softmax / Argmax
                                                       │
                                                Predicted Token y_t
```

---

## 📐 Mathematical Formulation

### 1. Bahdanau (Additive) Attention
At decoding step $t$, given spatial feature slices $f_i$ ($i \in \{1, \dots, 64\}$) and previous decoder hidden state $h_{t-1}$:

$$\text{score}(f_i, h_{t-1}) = v_a^\top \tanh(W_a f_i + U_a h_{t-1})$$

$$\alpha_{t, i} = \frac{\exp(\text{score}(f_i, h_{t-1}))}{\sum_{k=1}^{64} \exp(\text{score}(f_k, h_{t-1}))}$$

$$\text{context}_t = \sum_{i=1}^{64} \alpha_{t, i} f_i$$

### 2. Decoder Recurrent Step
The context vector $\text{context}_t$ and embedded word vector $e(y_{t-1})$ are concatenated:

$$x_t = [\text{context}_t \,;\, e(y_{t-1})]$$

$$h_t, c_t = \text{LSTM}(x_t, (h_{t-1}, c_{t-1}))$$

$$P(y_t \mid y_{<t}, I) = \text{softmax}(W_v \text{ReLU}(W_d h_t))$$

### 3. Masked Loss Normalization
Padding tokens (`0` = `<pad>`) are strictly masked out so they contribute neither loss nor gradients:

$$\mathcal{L} = \frac{\sum_{i=1}^{N} \sum_{t=1}^{T} m_{i, t} \cdot \text{SCE}(y_{i, t}, \hat{y}_{i, t})}{\max\left(\sum_{i=1}^{N} \sum_{t=1}^{T} m_{i, t}, 1\right)}$$

where $m_{i, t} = \mathbb{I}(y_{i, t} \neq 0)$ and $\text{SCE}$ is Sparse Categorical Cross-Entropy.

---

## 📂 Project Structure

```text
V2_Attention/
├── app.py                      # Interactive Streamlit application
├── requirements.txt            # Dependency specifications
├── README.md                   # Documentation and usage guide
├── models/                     # Historical model artifacts
│   ├── tokenizer.json          # Trained vocabulary mapping (2,827 tokens)
│   └── v2_attention.weights.h5 # Historical trained weights
├── experiments/                # Output directory for new retraining runs
│   └── v2_canonical/
├── plots/                      # Loss curve plots
└── src/
    ├── attention.py            # Bahdanau attention layer
    ├── cnn_features.py         # Lazy InceptionV3 spatial feature extractor & unified loader
    ├── convert_features.py     # Safe streaming offline converter for memory-mapped storage
    ├── dataset.py              # Caption parsing and text cleaning
    ├── decoder.py              # LSTM decoder with attention integration
    ├── evaluate.py             # Leak-free streaming BLEU-1 to BLEU-4 evaluation
    ├── feature_store.py        # MmapFeatureStore disk-backed memory-mapped feature store
    ├── generate_caption.py     # Greedy decoding and inference pipeline
    ├── loss.py                 # Masked cross-entropy with valid-token normalization
    ├── model.py                # Full ImageCaptionModel container
    ├── plot_history.py         # Training loss visualization utility
    ├── preprocess.py           # Text preprocessing with single-letter preservation
    ├── save_tokenizer.py       # Deterministic tokenizer serialization
    ├── sequences.py            # Safe sequence padding and truncation
    ├── tokenizer.py            # Word-to-index mapping generator
    ├── training_data.py        # Streaming TF data pipeline
    ├── train_step.py           # GradientTape custom training step
    ├── train.py                # Auditable training entry point with safety guards
    └── vocabulary.py           # Frequency-filtered vocabulary builder
```

---

## 🛠️ Usage Instructions

### 1. Verification and Dry Run
Non-mutating validation-only dry-run (validates hyperparameters, split manifests, caption parsing, and vocabulary in memory without GPU checks, model training, or writing/overwriting any files):
```bash
python V2_Attention/src/train.py --dry_run
```

### 2. Controlled Model Training & Reproducibility
Train on the leak-free canonical 70/15/15 dataset split (artifacts save to `experiments/v2_canonical/models/v2_attention.weights.h5` to protect historical weights):

- **Reproducibility Seeding (`--seed`, default: 42):** Sets random seeds for Python `random`, `numpy.random`, and `tensorflow`. Controls layer initialization, dropout masks, and dataset sequence shuffling.
  > [!NOTE]
  > *Limits of Reproducibility:* Algorithmic seeds guarantee repeatable pseudorandom sequences across Python, NumPy, and TensorFlow. However, exact bit-for-bit numerical reproducibility across heterogeneous GPU hardware, CUDA/cuDNN versions, or multithreaded CPU operations (e.g. parallel reductions in oneDNN) cannot be guaranteed due to non-associative floating-point operations. Split membership is strictly invariant to seed values and is read directly from `splits.json`. Validation evaluation is deterministic as `val_dataset` sets `shuffle=False`.
- **Atomic History Persistence:** `history.json` is updated atomically after every completed epoch via temporary swap files (`.tmp` + `os.replace`), protecting completed epoch logs if execution is interrupted.
- **Synchronized Checkpoint Tracking:** The best checkpoint is chosen by minimizing valid-token masked `val_loss`. When a new minimum is achieved, its SHA-256 fingerprint, epoch count, and validation loss are atomically written into `config.json`.
- **Experiment Protection (`--allow_overwrite_experiment`):** Aborts execution if target directory contains a completed experiment unless explicitly passed.

```bash
python V2_Attention/src/train.py \
    --split canonical \
    --output_dir V2_Attention/experiments/v2_canonical \
    --features_file V2_Attention/model/v2_features.npy \
    --seed 42 \
    --epochs 40 \
    --batch_size 32
```

### 3. Evaluation (Corpus BLEU-1 to BLEU-4)
Evaluate on held-out validation or test partitions across the entire dataset via `evaluate.py`:

> [!IMPORTANT]
> **Data Leakage Notice:** Historical weights `v2_attention.weights.h5` were trained on `historical_v2_split["train"]`, which contains 100% of the images in `canonical_split["test"]` (1,215 images) and `canonical_split["val"]` (1,213 images). Attempting to evaluate historical weights on `canonical_test` constitutes 100% training data leakage and is blocked by the leak guard. Historical weights must ONLY be evaluated on `historical_v2_val`. Canonical test evaluation is reserved exclusively for newly trained canonical models.

```bash
# Evaluate historical model on matching historical validation split (Valid):
python V2_Attention/src/evaluate.py --split historical_v2_val

# Evaluate a newly retrained canonical model on the canonical test split:
python V2_Attention/src/evaluate.py \
    --split canonical_test \
    --trained_on canonical \
    --weights V2_Attention/experiments/v2_canonical/models/v2_attention.weights.h5 \
    --tokenizer V2_Attention/experiments/v2_canonical/models/tokenizer.json
```

### 4. Single-Image Sample Inference
To run greedy inference with spatial attention on an individual image:
```bash
python V2_Attention/src/generate_caption.py
```
> [!NOTE]
> `generate_caption.py` generates captions and attention weights for qualitative inspection of individual sample images, whereas `evaluate.py` performs rigorous corpus-level BLEU-1 through BLEU-4 benchmarking across complete held-out partitions.

### 5. Visualizing Loss Curves
```bash
python V2_Attention/src/plot_history.py \
    --history V2_Attention/experiments/v2_canonical/history.json \
    --output_dir V2_Attention/plots
```

### 6. Running the Web Application
Launch from the repository root:
```bash
streamlit run V2_Attention/app.py
```
Or launch from inside `V2_Attention`:
```bash
streamlit run app.py
```

---

## 💾 Memory-Efficient Feature Storage (`MmapFeatureStore`)

To eliminate high RAM usage during training and evaluation, V2 supports a disk-backed, memory-mapped feature store alongside the legacy pickled dictionary format.

### 1. Legacy vs. Memory-Mapped Storage Comparison

| Dimension | Legacy Format (`v2_features.npy`) | Memory-Mapped Format (`MmapFeatureStore`) |
| :--- | :--- | :--- |
| **Physical File Layout** | Single pickled Python dictionary file | Directory with `manifest.json` + `features.npy` |
| **Pickle Dependency** | Required (`allow_pickle=True`), arbitrary object deserialization | Strictly prohibited (`allow_pickle=False`), pure IEEE 754 float32 |
| **Startup RAM Overhead** | ~4.2 GB allocated immediately to hold 8,091 arrays & dict wrappers | ~250 KB (only the lightweight JSON manifest index is loaded) |
| **Runtime Working Set** | Entire dataset remains pinned in process memory | Zero dataset pinning; only active batch tensors exist in RAM |
| **Operating System Paging** | None; all memory resides in private process heap | Leverages OS page cache; pages in slices on demand |
| **Access Time** | Fast dictionary lookup in RAM | Fast (O(1) dict lookup + memory-mapped pointer slice) |

### 2. Manifest Schema and File Layout

A memory-mapped feature store directory (e.g. `V2_Attention/model/v2_features_mmap/`) contains:
- `features.npy`: A contiguous binary NumPy array file of shape `(N, 64, 2048)` with data type `float32`.
- `manifest.json`: A metadata manifest containing index pointers:
  ```json
  {
    "format_version": "2.0",
    "storage_type": "mmap_numeric_array",
    "array_file": "features.npy",
    "num_images": 8091,
    "feature_shape": [64, 2048],
    "full_shape": [8091, 64, 2048],
    "dtype": "float32",
    "image_to_idx": {
      "1000268201_693b08cb0e.jpg": 0,
      "1001773457_577c3a7d70.jpg": 1
    }
  }
  ```

### 3. Safe Offline Feature Conversion

The offline converter (`V2_Attention/src/convert_features.py`) migrates a legacy dictionary `.npy` file into a memory-mapped store without allocating duplicate dataset arrays:

```bash
python V2_Attention/src/convert_features.py \
    --input V2_Attention/model/v2_features.npy \
    --output V2_Attention/model/v2_features_mmap/
```

> [!IMPORTANT]
> **Audit Phase Restriction:** The conversion command above must **NOT** be executed during audit phases. The historical feature file `V2_Attention/model/v2_features.npy` must remain unmodified. This tool is provided for user-directed offline migration when ready.

#### Converter Architecture & Safety Guarantees:
- **No Duplicate In-Memory Dataset:** Rather than creating a second full NumPy array in memory, the converter directly opens a disk-backed array using `np.lib.format.open_memmap(..., mode='w+')` and streams each image feature directly into disk storage row-by-row.
- **Strict Shape Normalization:** Per-image arrays with shapes `(64, 2048)`, `(8, 8, 2048)`, or `(1, 64, 2048)` are automatically validated and normalized to `(64, 2048)`. Irregular or mismatched shapes raise a clear `ValueError`.
- **Numeric Validation:** Every feature array is validated for finiteness via `np.all(np.isfinite(...))`. Arrays containing `NaN` or `Inf` are rejected immediately.
- **Staging Directory & Atomic Promotion:** Output files are written into a temporary staging folder (`.tmp_convert_*`). On successful completion, the staging directory is atomically promoted to the final destination. If an error occurs, the temporary directory is cleaned up and all file handles are closed, preventing partial or corrupted stores.
- **Overwrite & Replacement Safety:** The converter refuses to overwrite the input file. Default behavior strictly refuses overwrites if destination exists and is non-empty. On Windows, because atomic directory replacement cannot be guaranteed due to OS file locking semantics, replacing an existing populated directory is refused to guarantee existing data preservation, requiring a new destination directory.

### 4. Honest Memory Characteristics & Performance Trade-offs

- **Peak Memory During Conversion:**
  Converting a legacy pickled dictionary requires loading the original `.npy` file once into RAM (~4.2 GB heap memory). Peak memory during conversion is therefore approximately 4.2 GB + minor Python overhead. However, because destination rows are written sequentially into `open_memmap`, the converter does not require 2x dataset size (~8.4 GB).
- **Peak Memory During Training & Inference:**
  Using `MmapFeatureStore` during training or evaluation requires only ~250 KB for the JSON index in Python memory. As batches are loaded via `tf.data`, only the active batch tensor (`batch_size x 64 x 2048`, ~16 MB for batch size 32) is allocated in RAM.
- **Disk I/O and OS Page Caching:**
  Memory mapping relies on the OS virtual memory manager and page cache. Sequential or shuffled batch access reads slices directly from disk cache. On modern SSDs, throughput is virtually indistinguishable from heap-resident arrays.
- **Transparent Backward Compatibility:**
  `V2_Attention/src/cnn_features.py:load_features()` automatically detects whether the input path is a legacy `.npy` dictionary, an `.npz` archive, a directory containing `manifest.json`, or a path pointing directly to `manifest.json`. Existing workflows require zero code modifications.

---

## ⚖️ Fair Comparison with V1 Baseline

- **Target and Loss Semantics:** Both V1 and V2 optimize cross-entropy exclusively across valid non-padding tokens. In V1, the data generator outputs prefix-target pairs where each target is a single valid caption word ($w_i \in \{1, \dots, V-1\}$) and `<pad>` never appears as a target. In V2, full sequence targets are evaluated with dynamic masking that zeroes out padding positions ($m_{i,t} = 0$) and normalizes batch loss by the count of valid tokens. Padding tokens do not pull down the loss in either model.
- **Fair Comparison Criterion:** Because cross-entropy measures single-step predictive probability under teacher forcing with ground-truth prefixes, training and validation loss curves cannot measure compound error propagation during autoregressive generation. Cross-architecture benchmarking must compare Corpus BLEU (BLEU-1 through BLEU-4) measured on the mutually held-out canonical test split (`canonical_split["test"]`) using identical reference captions.

---

## 🔒 Safety and Integrity Guards

- **Protection Against Overwrite:** Historical checkpoints in `models/` cannot be overwritten without passing `--allow_overwrite_historical`. Existing completed experiments cannot be overwritten without `--allow_overwrite_experiment`.
- **Zero InceptionV3 Eager Import Overhead:** CNN weights are lazily loaded via `_LazyCNNProxy`, ensuring scripts and tests import in milliseconds.
- **Leak-Free Splits:** Pre-evaluation guards verify that no image in the evaluation set exists in the training set.
