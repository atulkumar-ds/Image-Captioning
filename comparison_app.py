"""
CaptionLab — Unified Image Captioning Model Comparison
=====================================================
A premium side-by-side comparison web application for evaluating vision-language models:
- V1: Baseline CNN (InceptionV3 Global Pooling) + LSTM Decoder
- V2: InceptionV3 Spatial Feature Map (8x8x2048) + Bahdanau Additive Attention + LSTM/GRU Decoder

Designed for native Windows execution and Streamlit Community Cloud readiness.
"""

import os
import sys
import json
import time
import tempfile
from pathlib import Path
from datetime import datetime
from PIL import Image

import streamlit as st

# =============================================================================
# Path Setup & Package Anchorage
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure V2 src is in sys.path so model.py can locate encoder and decoder
V2_SRC = REPO_ROOT / "V2_Attention" / "src"
if str(V2_SRC) not in sys.path:
    sys.path.insert(0, str(V2_SRC))

# Import existing V1 and V2 inference pipelines without modifying originals
import V1_Baseline_CNN_LSTM.src.cnn_features as v1_cnn_module
import V1_Baseline_CNN_LSTM.src.generate_caption as v1_caption_module
import V2_Attention.src.generate_caption as v2_caption_module

# =============================================================================
# Page Configuration & Refined Dark Theme Styling
# =============================================================================

st.set_page_config(
    page_title="CaptionLab — Image Captioning Model Comparison",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
/* Main app canvas */
.stApp {
    background-color: #0d1117;
    color: #e6edf3;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Header Container */
.captionlab-header {
    background: linear-gradient(135deg, rgba(22, 27, 34, 0.95) 0%, rgba(13, 17, 23, 0.95) 100%);
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 24px;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.35);
}

.captionlab-title-row {
    display: flex;
    align-items: center;
    gap: 14px;
    flex-wrap: wrap;
    margin-bottom: 6px;
}

.captionlab-title {
    font-size: 28px;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.5px;
    margin: 0;
}

.captionlab-badge {
    background: rgba(56, 139, 253, 0.15);
    color: #58a6ff;
    border: 1px solid rgba(56, 139, 253, 0.35);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.3px;
    text-transform: uppercase;
}

.captionlab-subtitle {
    color: #8b949e;
    font-size: 14px;
    margin: 4px 0 0 0;
    line-height: 1.5;
}

/* Upload Card */
.upload-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 24px;
}

/* Model Card Headers */
.model-card-v1 {
    background: #161b22;
    border: 1px solid #1f6feb;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 16px rgba(31, 111, 235, 0.12);
}

.model-card-v2 {
    background: #161b22;
    border: 1px solid #8957e5;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 16px rgba(137, 87, 229, 0.12);
}

.model-badge-v1 {
    background: rgba(31, 111, 235, 0.2);
    color: #58a6ff;
    border: 1px solid rgba(31, 111, 235, 0.4);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 700;
    display: inline-block;
    margin-bottom: 8px;
}

.model-badge-v2 {
    background: rgba(137, 87, 229, 0.2);
    color: #bc8cff;
    border: 1px solid rgba(137, 87, 229, 0.4);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 700;
    display: inline-block;
    margin-bottom: 8px;
}

.caption-box {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 16px;
    margin: 12px 0;
    font-size: 16px;
    line-height: 1.5;
    color: #f0f6fc;
}

.caption-quote {
    font-size: 17px;
    font-weight: 600;
    color: #ffffff;
    font-style: italic;
}

.meta-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #21262d;
    color: #8b949e;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 12px;
    margin-right: 6px;
    margin-bottom: 6px;
}

.meta-pill strong {
    color: #c9d1d9;
}

/* Comparison Insights Card */
.insights-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px;
    margin-top: 24px;
}

.metric-pill-faster {
    background: rgba(46, 160, 67, 0.15);
    color: #3fb950;
    border: 1px solid rgba(46, 160, 67, 0.35);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
}

.token-shared {
    background: rgba(56, 139, 253, 0.15);
    color: #58a6ff;
    border: 1px solid rgba(56, 139, 253, 0.3);
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 12px;
    font-family: monospace;
    display: inline-block;
    margin: 2px;
}

.token-diff-v1 {
    background: rgba(31, 111, 235, 0.15);
    color: #79c0ff;
    border: 1px solid rgba(31, 111, 235, 0.3);
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 12px;
    font-family: monospace;
    display: inline-block;
    margin: 2px;
}

.token-diff-v2 {
    background: rgba(137, 87, 229, 0.15);
    color: #d2a8ff;
    border: 1px solid rgba(137, 87, 229, 0.3);
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 12px;
    font-family: monospace;
    display: inline-block;
    margin: 2px;
}

.token-unk {
    background: rgba(210, 153, 34, 0.2);
    color: #e3b341;
    border: 1px solid rgba(210, 153, 34, 0.4);
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 12px;
    font-family: monospace;
    display: inline-block;
    margin: 2px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =============================================================================
# Artifact Resolution (Dynamic with Environment Overrides & Legacy Fallbacks)
# =============================================================================

def resolve_v1_paths():
    """Resolves V1 model checkpoint and tokenizer file paths."""
    env_model = os.environ.get("V1_MODEL_PATH")
    env_tokenizer = os.environ.get("V1_TOKENIZER_PATH")

    exp_dir = REPO_ROOT / "V1_Baseline_CNN_LSTM" / "experiments" / "v1_gpu_retrain_20261027" / "models"
    exp_model = exp_dir / "best_caption_model.keras"
    exp_tokenizer = exp_dir / "tokenizer.json"

    legacy_dir = REPO_ROOT / "V1_Baseline_CNN_LSTM" / "models"
    legacy_model = legacy_dir / "best_caption_model.keras"
    legacy_tokenizer = legacy_dir / "tokenizer.json"

    model_p = Path(env_model).resolve() if env_model else (exp_model if exp_model.exists() else legacy_model)
    tok_p = Path(env_tokenizer).resolve() if env_tokenizer else (exp_tokenizer if exp_tokenizer.exists() else legacy_tokenizer)

    return model_p, tok_p


def resolve_v2_paths():
    """Resolves V2 weights checkpoint and tokenizer file paths."""
    env_weights = os.environ.get("V2_WEIGHTS_PATH") or os.environ.get("V2_MODEL_PATH")
    env_tokenizer = os.environ.get("V2_TOKENIZER_PATH")

    exp_dir = REPO_ROOT / "V2_Attention" / "experiments" / "v2_attention_gpu_retrain_20261027" / "models"
    exp_weights = exp_dir / "v2_attention.weights.h5"
    exp_tokenizer = exp_dir / "tokenizer.json"

    legacy_dir = REPO_ROOT / "V2_Attention" / "models"
    legacy_weights = legacy_dir / "v2_attention.weights.h5"
    legacy_tokenizer = legacy_dir / "tokenizer.json"

    weights_p = Path(env_weights).resolve() if env_weights else (exp_weights if exp_weights.exists() else legacy_weights)
    tok_p = Path(env_tokenizer).resolve() if env_tokenizer else (exp_tokenizer if exp_tokenizer.exists() else legacy_tokenizer)

    return weights_p, tok_p


V1_MODEL_FILE, V1_TOKENIZER_FILE = resolve_v1_paths()
V2_WEIGHTS_FILE, V2_TOKENIZER_FILE = resolve_v2_paths()

# =============================================================================
# Cached Resource Pipelines
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_v1_pipeline(model_path_str: str, tokenizer_path_str: str):
    """Loads and caches V1 tokenizer, InceptionV3 global feature extractor, and LSTM decoder."""
    model_p = Path(model_path_str)
    tok_p = Path(tokenizer_path_str)

    if not tok_p.exists():
        raise FileNotFoundError(f"V1 Tokenizer not found at: {tok_p}")
    if not model_p.exists():
        raise FileNotFoundError(f"V1 Model checkpoint not found at: {model_p}")

    word_to_index, index_to_word, max_length = v1_caption_module.load_tokenizer(str(tok_p))
    model = v1_caption_module.load_caption_model(
        model_file=str(model_p),
        vocab_size=len(word_to_index),
        max_length=max_length
    )
    cnn_model = v1_cnn_module.cnn_model

    return {
        "model": model,
        "word_to_index": word_to_index,
        "index_to_word": index_to_word,
        "max_length": max_length,
        "cnn": cnn_model,
        "checkpoint": model_p,
        "tokenizer": tok_p
    }


@st.cache_resource(show_spinner=False)
def load_v2_pipeline(weights_path_str: str, tokenizer_path_str: str):
    """Loads and caches V2 tokenizer, InceptionV3 spatial feature extractor, and Attention decoder."""
    weights_p = Path(weights_path_str)
    tok_p = Path(tokenizer_path_str)

    if not tok_p.exists():
        raise FileNotFoundError(f"V2 Tokenizer not found at: {tok_p}")
    if not weights_p.exists():
        raise FileNotFoundError(f"V2 Model weights not found at: {weights_p}")

    word_to_index, index_to_word, max_length = v2_caption_module.load_tokenizer(str(tok_p))
    cnn = v2_caption_module.load_cnn()
    model = v2_caption_module.load_caption_model(
        vocab_size=len(word_to_index),
        weights_file=str(weights_p)
    )

    return {
        "model": model,
        "cnn": cnn,
        "word_to_index": word_to_index,
        "index_to_word": index_to_word,
        "max_length": max_length,
        "weights": weights_p,
        "tokenizer": tok_p
    }

# =============================================================================
# Helper Utilities
# =============================================================================

def format_relative_path(path: Path) -> str:
    """Returns a clean path relative to repo root for UI display."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return path.name


def analyze_vocabulary_comparison(cap_v1: str, cap_v2: str):
    """Performs token-level comparison between the two predictions."""
    tokens_v1 = [w.strip() for w in cap_v1.split() if w.strip()]
    tokens_v2 = [w.strip() for w in cap_v2.split() if w.strip()]

    set_v1 = set(tokens_v1)
    set_v2 = set(tokens_v2)

    shared = sorted(list(set_v1.intersection(set_v2)))
    v1_only = [w for w in tokens_v1 if w not in set_v2]
    v2_only = [w for w in tokens_v2 if w not in set_v1]

    union_len = len(set_v1.union(set_v2))
    jaccard_sim = (len(shared) / union_len) if union_len > 0 else 0.0

    return {
        "tokens_v1": tokens_v1,
        "tokens_v2": tokens_v2,
        "shared": shared,
        "v1_only": v1_only,
        "v2_only": v2_only,
        "jaccard_similarity": jaccard_sim,
        "has_unk_v1": "<unk>" in tokens_v1,
        "has_unk_v2": "<unk>" in tokens_v2
    }

# =============================================================================
# Sidebar Diagnostic & Configuration Panel
# =============================================================================

with st.sidebar:
    st.markdown("### 🔬 CaptionLab Control")
    st.markdown(
        "A rigorous side-by-side evaluation benchmark for comparing the baseline "
        "and attention architectures on identical input visual features."
    )

    st.markdown("---")
    st.markdown("#### Active Model Artifacts")

    st.markdown(
        f"""
**V1 Baseline Checkpoint:**  
`{format_relative_path(V1_MODEL_FILE)}`  
**V1 Tokenizer:**  
`{format_relative_path(V1_TOKENIZER_FILE)}`  

---

**V2 Attention Weights:**  
`{format_relative_path(V2_WEIGHTS_FILE)}`  
**V2 Tokenizer:**  
`{format_relative_path(V2_TOKENIZER_FILE)}`  
"""
    )

    st.markdown("---")
    st.markdown("#### Hardware & Environment")
    st.markdown(
        """
- **Runtime:** Native Windows (PowerShell)
- **Engine:** TensorFlow 2.21.0 / Keras 3.14.1
- **Device:** CPU Mode (Optimized)
- **ImageNet Encoder:** InceptionV3
"""
    )

    st.markdown("---")
    st.markdown("#### Standalone App Ports")
    st.markdown(
        """
- **V1 App:** `http://localhost:8502`
- **V2 App:** `http://localhost:8503`
- **Comparison App:** Current port
"""
    )

# =============================================================================
# Application Header
# =============================================================================

st.markdown(
    """
<div class="captionlab-header">
    <div class="captionlab-title-row">
        <h1 class="captionlab-title">🔬 CaptionLab</h1>
        <span class="captionlab-badge">2 Models Active</span>
    </div>
    <p class="captionlab-subtitle">
        Unified side-by-side comparative evaluation of vision-language architectures.
        Upload an image to benchmark global CNN+LSTM versus fine-grained Bahdanau Spatial Attention in real time.
    </p>
</div>
""",
    unsafe_allow_html=True
)

# =============================================================================
# Staged Pipeline Initialization
# =============================================================================

v1_pipeline = None
v2_pipeline = None
v1_load_error = None
v2_load_error = None

col_init1, col_init2 = st.columns(2)

with col_init1:
    try:
        v1_pipeline = load_v1_pipeline(str(V1_MODEL_FILE), str(V1_TOKENIZER_FILE))
    except Exception as e:
        v1_load_error = str(e)
        st.error(f"**V1 Initialization Failed:** {e}")

with col_init2:
    try:
        v2_pipeline = load_v2_pipeline(str(V2_WEIGHTS_FILE), str(V2_TOKENIZER_FILE))
    except Exception as e:
        v2_load_error = str(e)
        st.error(f"**V2 Initialization Failed:** {e}")

# =============================================================================
# Session State Initialization
# =============================================================================

if "comparison_result" not in st.session_state:
    st.session_state.comparison_result = None

if "last_image_key" not in st.session_state:
    st.session_state.last_image_key = None

# =============================================================================
# Main Image Upload Section
# =============================================================================

st.markdown("### 📤 Input Image Selection")

uploaded_file = st.file_uploader(
    label="Upload an image for comparative inference",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPEG, JPG, PNG. Both models evaluate on identical visual pixels.",
    label_visibility="collapsed"
)

# Image preview and metadata cards
if uploaded_file is not None:
    # Compute an identifier to detect image changes
    current_image_key = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.last_image_key != current_image_key:
        st.session_state.last_image_key = current_image_key
        st.session_state.comparison_result = None

    # Load image for preview & metadata
    try:
        pil_image = Image.open(uploaded_file)
        img_width, img_height = pil_image.size
        img_mode = pil_image.mode
        img_size_kb = uploaded_file.size / 1024
    except Exception as e:
        st.error(f"Failed to read image: {e}")
        st.stop()

    preview_col1, preview_col2 = st.columns([1, 2])

    with preview_col1:
        st.image(
            pil_image,
            caption=f"Uploaded: {uploaded_file.name}",
            use_container_width=True
        )

    with preview_col2:
        st.markdown(
            f"""
<div class="upload-card">
    <h4 style="margin: 0 0 12px 0; color: #ffffff;">Image Properties</h4>
    <div style="margin-bottom: 12px;">
        <span class="meta-pill"><strong>File Name:</strong> {uploaded_file.name}</span>
        <span class="meta-pill"><strong>Dimensions:</strong> {img_width} &times; {img_height} px</span>
        <span class="meta-pill"><strong>Size:</strong> {img_size_kb:.1f} KB</span>
        <span class="meta-pill"><strong>Color Space:</strong> {img_mode}</span>
    </div>
    <p style="color: #8b949e; font-size: 13px; margin: 0 0 16px 0;">
        Both architectures will resize this image to <strong>299 &times; 299</strong> with InceptionV3 preprocessing.
        Click <strong>Compare Models</strong> below to execute simultaneous inference.
    </p>
</div>
""",
            unsafe_allow_html=True
        )

        col_act1, col_act2 = st.columns([2, 1])
        with col_act1:
            compare_clicked = st.button(
                "🚀 Compare Models",
                type="primary",
                use_container_width=True,
                disabled=(v1_pipeline is None and v2_pipeline is None)
            )
        with col_act2:
            if st.button("🔄 Reset / Clear", use_container_width=True):
                st.session_state.comparison_result = None
                st.session_state.last_image_key = None
                st.rerun()

    # =========================================================================
    # Inference Execution on Compare Models Click
    # =========================================================================
    if compare_clicked:
        with st.spinner("Executing dual-model inference pipeline..."):
            # Save uploaded bytes to a safe temporary file for model ingestion
            temp_dir = REPO_ROOT / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_img_path = temp_dir / f"captionlab_{int(time.time()*1000)}_{uploaded_file.name}"

            with open(temp_img_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            v1_caption = None
            v1_latency = None
            v1_error = None

            v2_caption = None
            v2_latency = None
            v2_error = None

            # -------------------------------------------------------------
            # Execute V1 Inference
            # -------------------------------------------------------------
            if v1_pipeline is not None:
                try:
                    t_start_v1 = time.perf_counter()
                    v1_features = v1_cnn_module.extract_feature(
                        str(temp_img_path),
                        v1_pipeline["cnn"]
                    )
                    v1_caption = v1_caption_module.generate_caption(
                        model=v1_pipeline["model"],
                        image_feature=v1_features,
                        word_to_index=v1_pipeline["word_to_index"],
                        index_to_word=v1_pipeline["index_to_word"],
                        max_length=v1_pipeline["max_length"]
                    )
                    v1_latency = time.perf_counter() - t_start_v1
                except Exception as e:
                    v1_error = str(e)
            else:
                v1_error = v1_load_error or "V1 pipeline unavailable"

            # -------------------------------------------------------------
            # Execute V2 Inference
            # -------------------------------------------------------------
            if v2_pipeline is not None:
                try:
                    t_start_v2 = time.perf_counter()
                    v2_caption = v2_caption_module.generate_caption(
                        image_path=temp_img_path,
                        model=v2_pipeline["model"],
                        cnn=v2_pipeline["cnn"],
                        word_to_index=v2_pipeline["word_to_index"],
                        index_to_word=v2_pipeline["index_to_word"],
                        max_length=v2_pipeline["max_length"]
                    )
                    v2_latency = time.perf_counter() - t_start_v2
                except Exception as e:
                    v2_error = str(e)
            else:
                v2_error = v2_load_error or "V2 pipeline unavailable"

            # Clean up temporary file immediately after feature extraction
            if temp_img_path.exists():
                try:
                    os.remove(temp_img_path)
                except OSError:
                    pass

            # Store result in session state to persist through UI reruns
            st.session_state.comparison_result = {
                "timestamp": datetime.now().isoformat(),
                "image_name": uploaded_file.name,
                "image_width": img_width,
                "image_height": img_height,
                "image_size_kb": img_size_kb,
                "v1": {
                    "caption": v1_caption,
                    "latency": v1_latency,
                    "error": v1_error,
                    "checkpoint": format_relative_path(V1_MODEL_FILE),
                    "tokenizer": format_relative_path(V1_TOKENIZER_FILE),
                    "vocab_size": len(v1_pipeline["word_to_index"]) if v1_pipeline else 0,
                    "max_length": v1_pipeline["max_length"] if v1_pipeline else 0
                },
                "v2": {
                    "caption": v2_caption,
                    "latency": v2_latency,
                    "error": v2_error,
                    "weights": format_relative_path(V2_WEIGHTS_FILE),
                    "tokenizer": format_relative_path(V2_TOKENIZER_FILE),
                    "vocab_size": len(v2_pipeline["word_to_index"]) if v2_pipeline else 0,
                    "max_length": v2_pipeline["max_length"] if v2_pipeline else 0
                }
            }

else:
    # Empty State Guidance Card
    st.markdown(
        """
<div class="upload-card" style="text-align: center; padding: 48px 24px;">
    <div style="font-size: 42px; margin-bottom: 12px;">🖼️</div>
    <h3 style="color: #ffffff; margin-bottom: 8px;">No Image Selected</h3>
    <p style="color: #8b949e; max-width: 540px; margin: 0 auto 18px auto; font-size: 14px; line-height: 1.6;">
        Upload a picture above to initiate side-by-side inference. CaptionLab evaluates both the
        global-pooling CNN+LSTM baseline and the fine-grained spatial attention mechanism on identical pixels.
    </p>
    <div style="display: inline-flex; gap: 8px; flex-wrap: wrap; justify-content: center;">
        <span class="meta-pill">⚡ Real-time Inference</span>
        <span class="meta-pill">🔬 Token Alignment Analysis</span>
        <span class="meta-pill">⏱️ Monotonic Latency Benchmark</span>
    </div>
</div>
""",
        unsafe_allow_html=True
    )

# =============================================================================
# Side-by-Side Comparison Panel
# =============================================================================

res = st.session_state.comparison_result

if res is not None:
    st.markdown("---")
    st.markdown("### ⚖️ Side-by-Side Model Comparison")

    col_v1, col_v2 = st.columns(2)

    # -------------------------------------------------------------------------
    # Column 1: Model V1 (Baseline CNN + LSTM)
    # -------------------------------------------------------------------------
    with col_v1:
        st.markdown(
            f"""
<div class="model-card-v1">
    <span class="model-badge-v1">V1 BASELINE</span>
    <h3 style="margin: 0 0 6px 0; color: #ffffff;">CNN + LSTM Architecture</h3>
    <p style="color: #8b949e; font-size: 13px; margin: 0 0 10px 0;">
        InceptionV3 global average pooling (1 &times; 2048) &rarr; Embedding (256) &rarr; LSTM (256) &rarr; Dense Softmax.
    </p>
    <div style="margin-bottom: 12px;">
        <span class="meta-pill"><strong>Checkpoint:</strong> {res['v1']['checkpoint']}</span>
        <span class="meta-pill"><strong>Vocab:</strong> {res['v1']['vocab_size']}</span>
        <span class="meta-pill"><strong>Max Len:</strong> {res['v1']['max_length']}</span>
    </div>
</div>
""",
            unsafe_allow_html=True
        )

        v1_data = res["v1"]
        if v1_data["error"] is None and v1_data["caption"] is not None:
            clean_caption_v1 = v1_data["caption"].strip()
            display_text_v1 = clean_caption_v1 if clean_caption_v1 else "(Empty caption produced)"

            st.markdown(
                f"""
<div class="caption-box">
    <div style="font-size: 12px; color: #58a6ff; font-weight: 600; margin-bottom: 4px;">GENERATED CAPTION</div>
    <div class="caption-quote">"{display_text_v1}"</div>
</div>
""",
                unsafe_allow_html=True
            )

            # Special token notice
            if "<unk>" in clean_caption_v1:
                st.caption("⚠️ `<unk>` indicates an out-of-vocabulary word according to the 2,499-word training tokenizer.")

            # Performance stats
            lat_ms_v1 = v1_data["latency"] * 1000
            st.markdown(
                f"""
<div>
    <span class="meta-pill">⏱️ Latency: <strong>{lat_ms_v1:.1f} ms</strong> ({v1_data['latency']:.3f} s)</span>
    <span class="meta-pill">🔤 Word Count: <strong>{len(clean_caption_v1.split())}</strong></span>
</div>
""",
                unsafe_allow_html=True
            )

            # Copy caption helper
            st.code(clean_caption_v1, language=None)

        else:
            st.error(f"V1 Inference Failure: {v1_data['error']}")

    # -------------------------------------------------------------------------
    # Column 2: Model V2 (Spatial Attention + LSTM)
    # -------------------------------------------------------------------------
    with col_v2:
        st.markdown(
            f"""
<div class="model-card-v2">
    <span class="model-badge-v2">V2 ATTENTION</span>
    <h3 style="margin: 0 0 6px 0; color: #ffffff;">Spatial Attention Architecture</h3>
    <p style="color: #8b949e; font-size: 13px; margin: 0 0 10px 0;">
        InceptionV3 8&times;8 spatial feature grid (64 &times; 2048) &rarr; Bahdanau Additive Attention &rarr; GRU/LSTM (256).
    </p>
    <div style="margin-bottom: 12px;">
        <span class="meta-pill"><strong>Weights:</strong> {res['v2']['weights']}</span>
        <span class="meta-pill"><strong>Vocab:</strong> {res['v2']['vocab_size']}</span>
        <span class="meta-pill"><strong>Max Len:</strong> {res['v2']['max_length']}</span>
    </div>
</div>
""",
            unsafe_allow_html=True
        )

        v2_data = res["v2"]
        if v2_data["error"] is None and v2_data["caption"] is not None:
            clean_caption_v2 = v2_data["caption"].strip()
            display_text_v2 = clean_caption_v2 if clean_caption_v2 else "(Empty caption produced)"

            st.markdown(
                f"""
<div class="caption-box">
    <div style="font-size: 12px; color: #bc8cff; font-weight: 600; margin-bottom: 4px;">GENERATED CAPTION</div>
    <div class="caption-quote">"{display_text_v2}"</div>
</div>
""",
                unsafe_allow_html=True
            )

            # Special token notice
            if "<unk>" in clean_caption_v2:
                st.caption("⚠️ `<unk>` indicates an out-of-vocabulary word according to the 2,499-word training tokenizer.")

            # Performance stats
            lat_ms_v2 = v2_data["latency"] * 1000
            st.markdown(
                f"""
<div>
    <span class="meta-pill">⏱️ Latency: <strong>{lat_ms_v2:.1f} ms</strong> ({v2_data['latency']:.3f} s)</span>
    <span class="meta-pill">🔤 Word Count: <strong>{len(clean_caption_v2.split())}</strong></span>
</div>
""",
                unsafe_allow_html=True
            )

            # Copy caption helper
            st.code(clean_caption_v2, language=None)

        else:
            st.error(f"V2 Inference Failure: {v2_data['error']}")

    # =========================================================================
    # Comparison Insights & Quantitative Analysis
    # =========================================================================
    if (res["v1"]["error"] is None and res["v1"]["caption"] is not None and
        res["v2"]["error"] is None and res["v2"]["caption"] is not None):

        v1_cap = res["v1"]["caption"]
        v2_cap = res["v2"]["caption"]
        v1_lat = res["v1"]["latency"]
        v2_lat = res["v2"]["latency"]

        vocab_cmp = analyze_vocabulary_comparison(v1_cap, v2_cap)

        st.markdown("---")
        st.markdown("### 📊 Comparative Insights & Latency Analysis")

        ins_col1, ins_col2 = st.columns([1, 1])

        # Latency & Speed Comparison
        with ins_col1:
            st.markdown("#### ⚡ Latency Benchmark")
            abs_delta_ms = abs(v1_lat - v2_lat) * 1000

            if v1_lat < v2_lat:
                speed_ratio = v2_lat / v1_lat if v1_lat > 0 else 1.0
                faster_text = f"V1 Baseline completed {speed_ratio:.2f}&times; faster ({abs_delta_ms:.1f} ms delta)"
                badge_class = "metric-pill-faster"
            else:
                speed_ratio = v1_lat / v2_lat if v2_lat > 0 else 1.0
                faster_text = f"V2 Attention completed {speed_ratio:.2f}&times; faster ({abs_delta_ms:.1f} ms delta)"
                badge_class = "metric-pill-faster"

            st.markdown(
                f"""
<div class="upload-card">
    <div style="margin-bottom: 10px;">
        <span class="{badge_class}">⚡ {faster_text}</span>
    </div>
    <ul style="color: #c9d1d9; font-size: 13px; margin: 0; padding-left: 18px; line-height: 1.7;">
        <li><strong>V1 Latency:</strong> {v1_lat*1000:.1f} ms</li>
        <li><strong>V2 Latency:</strong> {v2_lat*1000:.1f} ms</li>
        <li><strong>Absolute Latency Delta:</strong> {abs_delta_ms:.1f} ms</li>
    </ul>
    <p style="color: #8b949e; font-size: 12px; margin-top: 10px; margin-bottom: 0;">
        <em>Note: V1 computes a single pooled 2048-dim feature vector per image, whereas V2 dynamic attention computes
        additive score projections over 64 spatial vectors for each decoded timestep.</em>
    </p>
</div>
""",
                unsafe_allow_html=True
            )

        # Token Alignment Comparison
        with ins_col2:
            st.markdown("#### 🔤 Vocabulary & Token Overlap")

            shared_html = " ".join([f"<span class='token-shared'>{w}</span>" for w in vocab_cmp["shared"]]) if vocab_cmp["shared"] else "<em>None</em>"
            v1_only_html = " ".join([f"<span class='token-diff-v1'>{w}</span>" for w in vocab_cmp["v1_only"]]) if vocab_cmp["v1_only"] else "<em>None</em>"
            v2_only_html = " ".join([f"<span class='token-diff-v2'>{w}</span>" for w in vocab_cmp["v2_only"]]) if vocab_cmp["v2_only"] else "<em>None</em>"

            st.markdown(
                f"""
<div class="upload-card">
    <div style="margin-bottom: 8px;">
        <span class="meta-pill">Jaccard Vocabulary Similarity: <strong>{vocab_cmp['jaccard_similarity']*100:.1f}%</strong></span>
    </div>
    <div style="font-size: 13px; line-height: 1.8; color: #c9d1d9;">
        <div><strong>Shared Tokens ({len(vocab_cmp['shared'])}):</strong> {shared_html}</div>
        <div style="margin-top: 6px;"><strong>V1-Only Tokens ({len(vocab_cmp['v1_only'])}):</strong> {v1_only_html}</div>
        <div style="margin-top: 6px;"><strong>V2-Only Tokens ({len(vocab_cmp['v2_only'])}):</strong> {v2_only_html}</div>
    </div>
    <p style="color: #8b949e; font-size: 12px; margin-top: 10px; margin-bottom: 0;">
        <em>Calculated strictly from actual generated strings without manual alterations.</em>
    </p>
</div>
""",
                unsafe_allow_html=True
            )

        # ---------------------------------------------------------------------
        # Report Downloads (JSON and Plain Text)
        # ---------------------------------------------------------------------
        st.markdown("#### 📥 Export Comparison Audit")

        report_json = json.dumps(
            {
                "captionlab_version": "1.0.0",
                "export_timestamp": datetime.now().isoformat(),
                "image": {
                    "filename": res["image_name"],
                    "width": res["image_width"],
                    "height": res["image_height"],
                    "size_kb": res["image_size_kb"]
                },
                "v1_baseline": {
                    "architecture": "InceptionV3_GlobalPooling_LSTM",
                    "checkpoint": res["v1"]["checkpoint"],
                    "caption": res["v1"]["caption"],
                    "latency_seconds": res["v1"]["latency"]
                },
                "v2_attention": {
                    "architecture": "InceptionV3_Spatial8x8_BahdanauAttention_LSTM",
                    "weights": res["v2"]["weights"],
                    "caption": res["v2"]["caption"],
                    "latency_seconds": res["v2"]["latency"]
                },
                "metrics": {
                    "latency_delta_ms": abs_delta_ms,
                    "jaccard_similarity": vocab_cmp["jaccard_similarity"],
                    "shared_tokens": vocab_cmp["shared"]
                }
            },
            indent=2
        )

        report_txt = f"""CAPTIONLAB MODEL COMPARISON REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Image: {res['image_name']} ({res['image_width']}x{res['image_height']} px, {res['image_size_kb']:.1f} KB)
============================================================
1. MODEL V1 (BASELINE CNN + LSTM)
   Checkpoint: {res['v1']['checkpoint']}
   Caption:    "{res['v1']['caption']}"
   Latency:    {res['v1']['latency']*1000:.1f} ms ({res['v1']['latency']:.3f} s)

2. MODEL V2 (SPATIAL ATTENTION)
   Weights:    {res['v2']['weights']}
   Caption:    "{res['v2']['caption']}"
   Latency:    {res['v2']['latency']*1000:.1f} ms ({res['v2']['latency']:.3f} s)

3. COMPARATIVE ANALYSIS
   Faster Model:        {'V1 Baseline' if v1_lat < v2_lat else 'V2 Attention'}
   Latency Delta:       {abs_delta_ms:.1f} ms
   Shared Words:        {', '.join(vocab_cmp['shared']) if vocab_cmp['shared'] else 'None'}
   Jaccard Similarity:  {vocab_cmp['jaccard_similarity']*100:.1f}%
============================================================
"""

        col_dl1, col_dl2 = st.columns([1, 1])
        with col_dl1:
            st.download_button(
                label="📄 Download Comparison Report (JSON)",
                data=report_json,
                file_name=f"captionlab_comparison_{Path(res['image_name']).stem}.json",
                mime="application/json",
                use_container_width=True
            )
        with col_dl2:
            st.download_button(
                label="📝 Download Comparison Summary (TXT)",
                data=report_txt,
                file_name=f"captionlab_comparison_{Path(res['image_name']).stem}.txt",
                mime="text/plain",
                use_container_width=True
            )
