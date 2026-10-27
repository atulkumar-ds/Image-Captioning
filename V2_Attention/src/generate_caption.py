import json
from pathlib import Path
import numpy as np
import tensorflow as tf

from keras.preprocessing import image
from keras.applications.inception_v3 import (
    InceptionV3,
    preprocess_input
)

try:
    from V2_Attention.src.model import ImageCaptionModel
except ImportError:
    from model import ImageCaptionModel


# ======================================================
# Configuration & Absolute Path Resolution
# ======================================================

SRC_DIR = Path(__file__).resolve().parent
V2_ROOT = SRC_DIR.parent

_EXP_TOKENIZER_FILE = V2_ROOT / "experiments" / "v2_attention_gpu_retrain_20261027" / "models" / "tokenizer.json"
_EXP_WEIGHTS_FILE = V2_ROOT / "experiments" / "v2_attention_gpu_retrain_20261027" / "models" / "v2_attention.weights.h5"
_LEGACY_TOKENIZER_FILE = V2_ROOT / "models" / "tokenizer.json"
_LEGACY_WEIGHTS_FILE = V2_ROOT / "models" / "v2_attention.weights.h5"

DEFAULT_TOKENIZER_FILE = _EXP_TOKENIZER_FILE if _EXP_TOKENIZER_FILE.exists() else _LEGACY_TOKENIZER_FILE
DEFAULT_WEIGHTS_FILE = _EXP_WEIGHTS_FILE if _EXP_WEIGHTS_FILE.exists() else _LEGACY_WEIGHTS_FILE

EMBEDDING_DIM = 256
UNITS = 256
MAX_LENGTH = 34


# ======================================================
# Load Tokenizer
# ======================================================

def load_tokenizer(tokenizer_file=None):

    if tokenizer_file is None:
        tokenizer_file = DEFAULT_TOKENIZER_FILE

    tokenizer_path = Path(tokenizer_file).resolve()
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer file not found: {tokenizer_path}"
        )

    with open(tokenizer_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    word_to_index = data["word_to_index"]

    index_to_word = {
        int(index): word
        for index, word in data["index_to_word"].items()
    }

    max_length = data.get(
        "max_length",
        MAX_LENGTH
    )

    return (
        word_to_index,
        index_to_word,
        max_length
    )


# ======================================================
# Load CNN
# ======================================================

def load_cnn():

    cnn = InceptionV3(
        weights="imagenet",
        include_top=False
    )

    cnn.trainable = False

    return cnn


# ======================================================
# Extract Features
# ======================================================

def extract_image_features(
    image_path,
    cnn
):

    image_path = Path(image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    img = image.load_img(
        str(image_path),
        target_size=(299, 299)
    )

    img = image.img_to_array(img)

    img = np.expand_dims(
        img,
        axis=0
    )

    img = preprocess_input(img)

    # Direct model invocation with training=False avoids repeated model.predict() overhead
    features = cnn(
        img,
        training=False
    ).numpy()

    # InceptionV3 output:
    # (1, 8, 8, 2048)
    #
    # Convert to:
    # (1, 64, 2048)

    features = features.reshape(
        1,
        -1,
        2048
    )

    return features.astype(
        np.float32
    )


# ======================================================
# Load Caption Model
# ======================================================

def load_caption_model(
    vocab_size,
    weights_file=None
):

    if weights_file is None:
        weights_file = DEFAULT_WEIGHTS_FILE

    weights_path = Path(weights_file).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights file not found: {weights_path}"
        )

    model = ImageCaptionModel(
        EMBEDDING_DIM,
        UNITS,
        vocab_size
    )

    # Verify model layer dimensions match tokenizer vocabulary
    assert model.decoder.embedding.input_dim == vocab_size, (
        f"Tokenizer vocab size ({vocab_size}) does not match "
        f"embedding input dimension ({model.decoder.embedding.input_dim})"
    )
    assert model.decoder.fc2.units == vocab_size, (
        f"Tokenizer vocab size ({vocab_size}) does not match "
        f"output layer units ({model.decoder.fc2.units})"
    )

    # Build subclassed model before loading weights
    dummy_features = tf.zeros(
        (1, 64, 2048),
        dtype=tf.float32
    )

    dummy_caption = tf.ones(
        (1, MAX_LENGTH),
        dtype=tf.int32
    )

    hidden, cell = model.decoder.reset_state(1)

    model(
        dummy_features,
        dummy_caption,
        hidden,
        cell
    )

    model.load_weights(
        str(weights_path)
    )

    return model


# ======================================================
# Generate Caption
# ======================================================

def generate_caption(
    image_path,
    model,
    cnn,
    word_to_index,
    index_to_word,
    max_length
):

    # --------------------------------------------------
    # Extract CNN features
    # --------------------------------------------------

    features = extract_image_features(
        image_path,
        cnn
    )

    features = tf.convert_to_tensor(
        features,
        dtype=tf.float32
    )

    # --------------------------------------------------
    # Encode image
    # --------------------------------------------------

    encoded_features = model.encoder(
        features
    )

    # --------------------------------------------------
    # Initialize decoder
    # --------------------------------------------------

    hidden, cell = model.decoder.reset_state(
        1
    )

    current_word = tf.constant(
        [[word_to_index["<start>"]]],
        dtype=tf.int32
    )

    generated_words = []

    # --------------------------------------------------
    # Generate one word at a time
    # --------------------------------------------------

    for _ in range(max_length - 1):

        prediction, hidden, cell, attention_weights = (
            model.decoder(
                current_word,
                encoded_features,
                hidden,
                cell
            )
        )

        predicted_id = tf.argmax(
            prediction[0]
        ).numpy()

        predicted_word = index_to_word.get(
            int(predicted_id),
            "<unk>"
        )

        # Stop when any special stopping token is predicted (<end>, <pad>, or <start>)
        if predicted_word in {"<end>", "<pad>", "<start>"}:
            break

        generated_words.append(
            predicted_word
        )

        current_word = tf.constant(
            [[predicted_id]],
            dtype=tf.int32
        )

    caption = " ".join(
        generated_words
    )

    return caption


# ======================================================
# Test
# ======================================================

if __name__ == "__main__":

    REPO_ROOT = V2_ROOT.parent
    IMAGE_PATH = (
        REPO_ROOT /
        "V1_Baseline_CNN_LSTM" /
        "data" /
        "images" /
        "1000268201_693b08cb0e.jpg"
    )

    word_to_index, index_to_word, max_length = (
        load_tokenizer(DEFAULT_TOKENIZER_FILE)
    )

    cnn = load_cnn()

    model = load_caption_model(
        len(word_to_index),
        DEFAULT_WEIGHTS_FILE
    )

    caption = generate_caption(
        IMAGE_PATH,
        model,
        cnn,
        word_to_index,
        index_to_word,
        max_length
    )

    print("\nGenerated Caption:")
    print(caption)
