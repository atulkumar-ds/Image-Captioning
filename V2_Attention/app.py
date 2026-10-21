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
def load_cached_resources():

    tokenizer_path = APP_DIR / "models" / "tokenizer.json"
    weights_path = APP_DIR / "models" / "v2_attention.weights.h5"

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
        tokenizer_path = APP_DIR / "models" / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

        st.write("2/3 Loading InceptionV3 CNN encoder...")
        weights_path = APP_DIR / "models" / "v2_attention.weights.h5"
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found at {weights_path}")

        st.write("3/3 Loading attention decoder weights...")
        (
            cnn,
            model,
            word_to_index,
            index_to_word,
            max_length
        ) = load_cached_resources()

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
        f"**App Directory:** `{APP_DIR}`"
    )
    st.stop()


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
