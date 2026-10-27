import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

# ======================================================
# Add src to Python path (anchored to app.py location)
# ======================================================

APP_DIR = Path(__file__).resolve().parent
V1_ROOT = APP_DIR.parent
SRC_DIR = V1_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cnn_features import (
    cnn_model,
    extract_feature
)

from generate_caption import (
    load_tokenizer,
    load_caption_model,
    generate_caption
)


# ======================================================
# Checkpoint and Tokenizer Resolution
# ======================================================

def resolve_v1_artifacts():
    """
    Resolves V1 model checkpoint and tokenizer file paths using pathlib.Path.

    Resolution order:
    1. Environment variables (V1_MODEL_PATH, V1_TOKENIZER_PATH) if provided.
    2. Trained experiment artifacts (preferred):
       V1_ROOT / 'experiments' / 'v1_gpu_retrain_20261027' / 'models' / ...
    3. Legacy models directory fallback:
       V1_ROOT / 'models' / ...
    """
    env_model = os.environ.get("V1_MODEL_PATH")
    env_tokenizer = os.environ.get("V1_TOKENIZER_PATH")

    exp_dir = V1_ROOT / "experiments" / "v1_gpu_retrain_20261027" / "models"
    exp_model = exp_dir / "best_caption_model.keras"
    exp_tokenizer = exp_dir / "tokenizer.json"

    legacy_dir = V1_ROOT / "models"
    legacy_model = legacy_dir / "best_caption_model.keras"
    legacy_tokenizer = legacy_dir / "tokenizer.json"

    # Resolve model checkpoint
    if env_model:
        model_file = Path(env_model).resolve()
    elif exp_model.exists():
        model_file = exp_model.resolve()
    elif legacy_model.exists():
        model_file = legacy_model.resolve()
    else:
        model_file = exp_model.resolve()

    # Resolve tokenizer
    if env_tokenizer:
        tokenizer_file = Path(env_tokenizer).resolve()
    elif model_file == exp_model.resolve() and exp_tokenizer.exists():
        tokenizer_file = exp_tokenizer.resolve()
    elif legacy_tokenizer.exists() and not env_tokenizer:
        tokenizer_file = legacy_tokenizer.resolve()
    else:
        tokenizer_file = exp_tokenizer.resolve()

    return model_file, tokenizer_file


MODEL_FILE, TOKENIZER_FILE = resolve_v1_artifacts()


st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="centered"
)


@st.cache_resource(show_spinner=False)
def load_resources(model_path_str: str, tokenizer_path_str: str):

    (
        word_to_index,
        index_to_word,
        max_length
    ) = load_tokenizer(
        tokenizer_path_str
    )

    model = load_caption_model(
        model_file=model_path_str,
        vocab_size=len(word_to_index),
        max_length=max_length
    )

    return (
        model,
        word_to_index,
        index_to_word,
        max_length
    )


# ======================================================
# Staged Loading with Error Handling
# ======================================================

model = None
word_to_index = None
index_to_word = None
max_length = None

try:
    with st.status("Initializing V1 Baseline Model...", expanded=False) as status:
        st.write("1/3 Verifying files and tokenizer...")
        if not TOKENIZER_FILE.exists():
            raise FileNotFoundError(
                f"Tokenizer file not found: {TOKENIZER_FILE}\n"
                f"Expected in experiment path: {V1_ROOT / 'experiments' / 'v1_gpu_retrain_20261027' / 'models' / 'tokenizer.json'}\n"
                f"Or legacy path: {V1_ROOT / 'models' / 'tokenizer.json'}\n"
                f"You can also set the V1_TOKENIZER_PATH environment variable."
            )
        if not MODEL_FILE.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {MODEL_FILE}\n"
                f"Expected in experiment path: {V1_ROOT / 'experiments' / 'v1_gpu_retrain_20261027' / 'models' / 'best_caption_model.keras'}\n"
                f"Or legacy path: {V1_ROOT / 'models' / 'best_caption_model.keras'}\n"
                f"You can also set the V1_MODEL_PATH environment variable."
            )

        st.write("2/3 Loading InceptionV3 CNN feature extractor...")
        # cnn_model is loaded via import

        st.write("3/3 Reconstructing V1 model & loading weights...")
        (
            model,
            word_to_index,
            index_to_word,
            max_length
        ) = load_resources(str(MODEL_FILE), str(TOKENIZER_FILE))

        status.update(
            label="✓ V1 Baseline Model ready",
            state="complete",
            expanded=False
        )

except Exception as e:
    st.error(
        f"**Model Initialization Failed**\n\n"
        f"**Error Type:** `{type(e).__name__}`\n\n"
        f"**Details:** {e}\n\n"
        f"**V1 Root:** `{V1_ROOT}`\n\n"
        f"**Resolved Checkpoint:** `{MODEL_FILE}`\n\n"
        f"**Resolved Tokenizer:** `{TOKENIZER_FILE}`"
    )
    st.stop()


st.title("🖼️ Image Caption Generator")

st.markdown(
    """
Generate captions for images using a
CNN + LSTM Deep Learning model.
"""
)


# Sidebar with model metadata and active checkpoint
st.sidebar.title("Model Information")

try:
    display_model_path = MODEL_FILE.relative_to(V1_ROOT)
except ValueError:
    display_model_path = MODEL_FILE.name

try:
    display_tok_path = TOKENIZER_FILE.relative_to(V1_ROOT)
except ValueError:
    display_tok_path = TOKENIZER_FILE.name

st.sidebar.markdown(
    f"""
### Architecture
- CNN: InceptionV3 (ImageNet)
- Decoder: Embedding + LSTM + Dense
- Dataset: Flickr8k

### Active Artifacts
- **Checkpoint:** `{display_model_path}`
- **Tokenizer:** `{display_tok_path}`
- **Vocabulary Size:** `{len(word_to_index)}`
- **Max Length:** `{max_length}`
"""
)


uploaded_image = st.file_uploader(
    "Upload an Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded_image is not None:

    st.image(
        uploaded_image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Generate Caption"):

        temp_dir = APP_DIR / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        temp_path = temp_dir / uploaded_image.name

        try:
            with open(temp_path, "wb") as f:
                f.write(uploaded_image.read())

            with st.spinner("Generating caption..."):
                image_feature = extract_feature(
                    str(temp_path),
                    cnn_model
                )

                caption = generate_caption(
                    model=model,
                    image_feature=image_feature,
                    word_to_index=word_to_index,
                    index_to_word=index_to_word,
                    max_length=max_length
                )

            st.success("Caption Generated Successfully!")
            st.subheader("Generated Caption")
            st.write(f"**{caption}**")
            st.balloons()

        except Exception as e:
            st.error(f"Error during caption generation: {e}")

        finally:
            if temp_path.exists():
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
