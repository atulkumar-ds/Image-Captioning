import sys
import os
from pathlib import Path

import streamlit as st


# ======================================================
# Add src to Python path (anchored to app.py location)
# ======================================================

APP_DIR = Path(__file__).resolve().parent
SRC_DIR = APP_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from generate_caption import (
    load_tokenizer,
    load_cnn,
    load_caption_model,
    generate_caption
)


# ======================================================
# Checkpoint and Tokenizer Resolution
# ======================================================

def resolve_v2_artifacts():
    """
    Resolves V2 weights checkpoint and tokenizer file paths using pathlib.Path.

    Resolution order:
    1. Environment variables (V2_WEIGHTS_PATH or V2_MODEL_PATH, V2_TOKENIZER_PATH) if provided.
    2. Newly trained experiment artifacts (preferred):
       APP_DIR / 'experiments' / 'v2_attention_gpu_retrain_20261027' / 'models' / ...
    3. Legacy models directory fallback:
       APP_DIR / 'models' / ...
    """
    env_weights = os.environ.get("V2_WEIGHTS_PATH") or os.environ.get("V2_MODEL_PATH")
    env_tokenizer = os.environ.get("V2_TOKENIZER_PATH")

    exp_dir = APP_DIR / "experiments" / "v2_attention_gpu_retrain_20261027" / "models"
    exp_weights = exp_dir / "v2_attention.weights.h5"
    exp_tokenizer = exp_dir / "tokenizer.json"

    legacy_dir = APP_DIR / "models"
    legacy_weights = legacy_dir / "v2_attention.weights.h5"
    legacy_tokenizer = legacy_dir / "tokenizer.json"

    # Resolve weights checkpoint
    if env_weights:
        weights_file = Path(env_weights).resolve()
    elif exp_weights.exists():
        weights_file = exp_weights.resolve()
    elif legacy_weights.exists():
        weights_file = legacy_weights.resolve()
    else:
        # Default target for clear error reporting
        weights_file = exp_weights.resolve()

    # Resolve tokenizer
    if env_tokenizer:
        tokenizer_file = Path(env_tokenizer).resolve()
    elif weights_file == exp_weights.resolve() and exp_tokenizer.exists():
        tokenizer_file = exp_tokenizer.resolve()
    elif legacy_tokenizer.exists() and not env_tokenizer:
        tokenizer_file = legacy_tokenizer.resolve()
    else:
        tokenizer_file = exp_tokenizer.resolve()

    return weights_file, tokenizer_file


WEIGHTS_FILE, TOKENIZER_FILE = resolve_v2_artifacts()


# ======================================================
# Page Configuration
# ======================================================

st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="🖼️",
    layout="centered"
)


# ======================================================
# Title
# ======================================================

st.title("🖼️ AI Image Caption Generator")

st.write(
    "Upload an image and let the V2 Attention "
    "model generate a caption."
)


# ======================================================
# Load Models (Cached)
# ======================================================

@st.cache_resource(show_spinner=False)
def load_cached_resources(weights_path_str: str, tokenizer_path_str: str):

    tokenizer_path = Path(tokenizer_path_str)
    weights_path = Path(weights_path_str)

    word_to_index, index_to_word, max_length = load_tokenizer(
        tokenizer_path
    )

    cnn = load_cnn()

    model = load_caption_model(
        vocab_size=len(word_to_index),
        weights_file=weights_path
    )

    return (
        cnn,
        model,
        word_to_index,
        index_to_word,
        max_length
    )


# ======================================================
# Staged Model Loading with Error Handling
# ======================================================

cnn = None
model = None
word_to_index = None
index_to_word = None
max_length = None

try:
    with st.status("Initializing V2 Attention Model...", expanded=False) as status:
        st.write("1/3 Loading tokenizer mappings...")
        if not TOKENIZER_FILE.exists():
            raise FileNotFoundError(
                f"Tokenizer not found at {TOKENIZER_FILE}\n"
                f"Expected in experiment path: {APP_DIR / 'experiments' / 'v2_attention_gpu_retrain_20261027' / 'models' / 'tokenizer.json'}\n"
                f"Or legacy path: {APP_DIR / 'models' / 'tokenizer.json'}\n"
                f"You can also set the V2_TOKENIZER_PATH environment variable."
            )

        st.write("2/3 Loading InceptionV3 CNN encoder...")
        if not WEIGHTS_FILE.exists():
            raise FileNotFoundError(
                f"Model weights not found at {WEIGHTS_FILE}\n"
                f"Expected in experiment path: {APP_DIR / 'experiments' / 'v2_attention_gpu_retrain_20261027' / 'models' / 'v2_attention.weights.h5'}\n"
                f"Or legacy path: {APP_DIR / 'models' / 'v2_attention.weights.h5'}\n"
                f"You can also set the V2_WEIGHTS_PATH environment variable."
            )

        st.write("3/3 Loading attention decoder weights...")
        (
            cnn,
            model,
            word_to_index,
            index_to_word,
            max_length
        ) = load_cached_resources(str(WEIGHTS_FILE), str(TOKENIZER_FILE))

        status.update(
            label="✓ V2 Attention Model ready",
            state="complete",
            expanded=False
        )

except Exception as e:
    st.error(
        f"**Model Initialization Failed**\n\n"
        f"**Error Type:** `{type(e).__name__}`\n\n"
        f"**Details:** {e}\n\n"
        f"**App Directory:** `{APP_DIR}`\n\n"
        f"**Resolved Weights:** `{WEIGHTS_FILE}`\n\n"
        f"**Resolved Tokenizer:** `{TOKENIZER_FILE}`"
    )
    st.stop()


# Sidebar with model metadata and active checkpoint
st.sidebar.title("Model Information")

try:
    display_weights_path = WEIGHTS_FILE.relative_to(APP_DIR)
except ValueError:
    display_weights_path = WEIGHTS_FILE.name

try:
    display_tok_path = TOKENIZER_FILE.relative_to(APP_DIR)
except ValueError:
    display_tok_path = TOKENIZER_FILE.name

st.sidebar.markdown(
    f"""
### Architecture
- Encoder: InceptionV3 (Spatial 8x8x2048)
- Attention: Bahdanau Additive Attention
- Decoder: GRU/LSTM with Attention Context
- Dataset: Flickr8k

### Active Artifacts
- **Weights:** `{display_weights_path}`
- **Tokenizer:** `{display_tok_path}`
- **Vocabulary Size:** `{len(word_to_index)}`
- **Max Length:** `{max_length}`
"""
)


# ======================================================
# Upload Image
# ======================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ======================================================
# Caption Generation
# ======================================================

if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button(
        "✨ Generate Caption",
        use_container_width=True
    ):

        temp_dir = APP_DIR / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        image_path = temp_dir / uploaded_file.name

        try:
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Generating caption..."):
                caption = generate_caption(
                    image_path,
                    model,
                    cnn,
                    word_to_index,
                    index_to_word,
                    max_length
                )

            st.subheader("Generated Caption")
            st.success(caption)

        except Exception as e:
            st.error(f"Error during caption generation: {e}")

        finally:
            if image_path.exists():
                try:
                    os.remove(image_path)
                except OSError:
                    pass
