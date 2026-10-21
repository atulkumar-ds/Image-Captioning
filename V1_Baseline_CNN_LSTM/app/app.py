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


MODEL_FILE = V1_ROOT / "models" / "best_caption_model.keras"
TOKENIZER_FILE = V1_ROOT / "models" / "tokenizer.json"


st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="centered"
)


@st.cache_resource(show_spinner=False)
def load_resources():

    (
        word_to_index,
        index_to_word,
        max_length
    ) = load_tokenizer(
        TOKENIZER_FILE
    )

    model = load_caption_model(
        model_file=MODEL_FILE,
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
            raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_FILE}")
        if not MODEL_FILE.exists():
            raise FileNotFoundError(f"Checkpoint not found: {MODEL_FILE}")

        st.write("2/3 Loading InceptionV3 CNN feature extractor...")
        # cnn_model is loaded via import

        st.write("3/3 Reconstructing V1 model & loading weights...")
        (
            model,
            word_to_index,
            index_to_word,
            max_length
        ) = load_resources()

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
        f"**V1 Root:** `{V1_ROOT}`"
    )
    st.stop()


st.title("🖼️ Image Caption Generator")

st.markdown(
    """
Generate captions for images using a
CNN + LSTM Deep Learning model.
"""
)


st.sidebar.title("Model Information")

st.sidebar.markdown(
    """
### CNN
- InceptionV3
- ImageNet Weights

### Decoder
- Embedding Layer
- LSTM
- Dense + Softmax

### Dataset
- Flickr8k
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
