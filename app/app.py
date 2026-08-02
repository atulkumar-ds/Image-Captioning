import os
import sys
import tempfile

import streamlit as st
from keras.models import load_model

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "src"
    )
)

from cnn_features import (
    cnn_model,
    extract_feature
)

from generate_caption import (
    load_tokenizer,
    generate_caption
)


MODEL_FILE = "models/best_caption_model.keras"
TOKENIZER_FILE = "models/tokenizer.json"


st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="centered"
)


@st.cache_resource
def load_resources():

    model = load_model(MODEL_FILE)

    (
        word_to_index,
        index_to_word,
        max_length
    ) = load_tokenizer(
        TOKENIZER_FILE
    )

    return (
        model,
        word_to_index,
        index_to_word,
        max_length
    )


(
    model,
    word_to_index,
    index_to_word,
    max_length
) = load_resources()


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

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as temp_file:

            temp_file.write(
                uploaded_image.read()
            )

            temp_path = temp_file.name

        with st.spinner(
            "Generating caption..."
        ):

            image_feature = extract_feature(
                temp_path,
                cnn_model
            )

            caption = generate_caption(
                model=model,
                image_feature=image_feature,
                word_to_index=word_to_index,
                index_to_word=index_to_word,
                max_length=max_length
            )

        os.remove(temp_path)

        st.success("Caption Generated Successfully!")

        st.subheader("Generated Caption")

        st.write(f"**{caption}**")

        st.balloons()