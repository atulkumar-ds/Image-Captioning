import sys
import os

import streamlit as st


# ======================================================
# Add src to Python path
# ======================================================

SRC_DIR = os.path.join(
    os.path.dirname(__file__),
    "src"
)

sys.path.insert(
    0,
    SRC_DIR
)


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
# Load Models
# ======================================================

@st.cache_resource
def load_models():

    tokenizer_path = os.path.join(
        os.path.dirname(__file__),
        "models",
        "tokenizer.json"
    )

    word_to_index, index_to_word, max_length = (
        load_tokenizer(
            tokenizer_path
        )
    )

    cnn = load_cnn()

    model = load_caption_model(
        len(word_to_index)
    )

    return (
        cnn,
        model,
        word_to_index,
        index_to_word,
        max_length
    )


# ======================================================
# Load
# ======================================================

with st.spinner("Loading V2 Attention model..."):

    (
        cnn,
        model,
        word_to_index,
        index_to_word,
        max_length
    ) = load_models()


st.success("Model loaded successfully!")


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

        temp_dir = "temp"

        os.makedirs(
            temp_dir,
            exist_ok=True
        )

        image_path = os.path.join(
            temp_dir,
            uploaded_file.name
        )

        with open(
            image_path,
            "wb"
        ) as f:

            f.write(
                uploaded_file.getbuffer()
            )

        with st.spinner(
            "Generating caption..."
        ):

            caption = generate_caption(
                image_path,
                model,
                cnn,
                word_to_index,
                index_to_word,
                max_length
            )

        st.subheader(
            "Generated Caption"
        )

        st.success(
            caption
        )

        # Remove temporary image

        try:
            os.remove(
                image_path
            )
        except:
            pass