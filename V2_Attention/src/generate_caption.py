import json
import numpy as np
import tensorflow as tf

from keras.preprocessing import image
from keras.applications.inception_v3 import (
    InceptionV3,
    preprocess_input
)

from model import ImageCaptionModel


# ======================================================
# Configuration
# ======================================================

EMBEDDING_DIM = 256
UNITS = 256

TOKENIZER_FILE = "models/tokenizer.json"
WEIGHTS_FILE = "models/v2_attention.weights.h5"

MAX_LENGTH = 34


# ======================================================
# Load Tokenizer
# ======================================================

def load_tokenizer(tokenizer_file):

    with open(tokenizer_file, "r") as f:
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

    img = image.load_img(
        image_path,
        target_size=(299, 299)
    )

    img = image.img_to_array(img)

    img = np.expand_dims(
        img,
        axis=0
    )

    img = preprocess_input(img)

    features = cnn.predict(
        img,
        verbose=0
    )

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
    vocab_size
):

    model = ImageCaptionModel(
        EMBEDDING_DIM,
        UNITS,
        vocab_size
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
        WEIGHTS_FILE
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

        if predicted_word == "<end>":
            break

        if predicted_word not in [
            "<pad>",
            "<start>"
        ]:

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

    IMAGE_PATH = (
        "../V1_Baseline_CNN_LSTM/"
        "data/images/"
        "1000268201_693b08cb0e.jpg"
    )

    word_to_index, index_to_word, max_length = (
        load_tokenizer(TOKENIZER_FILE)
    )

    cnn = load_cnn()

    model = load_caption_model(
        len(word_to_index)
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