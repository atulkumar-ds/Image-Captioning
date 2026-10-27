import json
from pathlib import Path
import numpy as np

from keras.preprocessing.sequence import pad_sequences

try:
    from V1_Baseline_CNN_LSTM.src.cnn_features import (
        cnn_model,
        extract_feature
    )
    from V1_Baseline_CNN_LSTM.src.model import build_caption_model
except ImportError:
    from cnn_features import (
        cnn_model,
        extract_feature
    )
    from model import build_caption_model


# ======================================================
# Paths & Absolute Resolution
# ======================================================

SRC_DIR = Path(__file__).resolve().parent
V1_ROOT = SRC_DIR.parent

_EXP_MODEL_FILE = V1_ROOT / "experiments" / "v1_gpu_retrain_20261027" / "models" / "best_caption_model.keras"
_EXP_TOKENIZER_FILE = V1_ROOT / "experiments" / "v1_gpu_retrain_20261027" / "models" / "tokenizer.json"
_LEGACY_MODEL_FILE = V1_ROOT / "models" / "best_caption_model.keras"
_LEGACY_TOKENIZER_FILE = V1_ROOT / "models" / "tokenizer.json"

DEFAULT_MODEL_FILE = _EXP_MODEL_FILE if _EXP_MODEL_FILE.exists() else _LEGACY_MODEL_FILE
DEFAULT_TOKENIZER_FILE = _EXP_TOKENIZER_FILE if _EXP_TOKENIZER_FILE.exists() else _LEGACY_TOKENIZER_FILE


# ======================================================
# Load Tokenizer
# ======================================================

def load_tokenizer(file_path=None):

    if file_path is None:
        file_path = DEFAULT_TOKENIZER_FILE

    tokenizer_path = Path(file_path).resolve()
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer file not found: {tokenizer_path}"
        )

    with open(
        tokenizer_path,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    word_to_index = data["word_to_index"]

    # JSON converts dictionary integer keys into strings.
    # Convert them back to integers.
    index_to_word = {
        int(index): word
        for index, word
        in data["index_to_word"].items()
    }

    max_length = data["max_length"]

    return (
        word_to_index,
        index_to_word,
        max_length
    )


# ======================================================
# Load Caption Model (Reconstruct + Load Weights)
# ======================================================

def load_caption_model(
    model_file=None,
    vocab_size=2814,
    max_length=34
):

    if model_file is None:
        model_file = DEFAULT_MODEL_FILE

    model_path = Path(model_file).resolve()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    # Reconstruct exact V1 architecture and load weights
    # to circumvent Keras whole-model deserialization config drift
    model = build_caption_model(
        vocab_size=vocab_size,
        max_length=max_length,
        feature_size=2048,
        embedding_dim=256,
        hidden_dim=256
    )

    model.load_weights(str(model_path))

    assert model.output_shape[-1] == vocab_size, (
        f"Output vocabulary size mismatch: {model.output_shape[-1]} != {vocab_size}"
    )

    return model


# ======================================================
# Generate Caption
# ======================================================

def generate_caption(
    model,
    image_feature,
    word_to_index,
    index_to_word,
    max_length
):

    # Start generation with <start>
    generated_words = ["<start>"]

    for _ in range(max_length):

        # Convert words -> indices
        sequence = [
            word_to_index.get(
                word,
                word_to_index["<unk>"]
            )
            for word in generated_words
        ]

        # Pad sequence
        sequence = pad_sequences(
            [sequence],
            maxlen=max_length,
            padding="post"
        )

        # Direct model invocation with training=False avoids repeated model.predict() overhead
        predictions = model(
            [
                image_feature.reshape(1, -1),
                sequence
            ],
            training=False
        ).numpy()

        # Word with highest probability
        predicted_index = np.argmax(
            predictions[0]
        )

        predicted_word = index_to_word.get(
            predicted_index,
            "<unk>"
        )

        # Stop if model predicts <end>
        if predicted_word == "<end>":
            break

        # Safety: don't generate padding/start tokens
        if predicted_word in {
            "<pad>",
            "<start>"
        }:
            break

        generated_words.append(
            predicted_word
        )

    # Remove <start>
    caption_words = generated_words[1:]

    return " ".join(caption_words)


# ======================================================
# Main
# ======================================================

if __name__ == "__main__":

    print("\nLoading tokenizer...")

    (
        word_to_index,
        index_to_word,
        max_length
    ) = load_tokenizer(
        DEFAULT_TOKENIZER_FILE
    )

    print("Tokenizer loaded successfully.")
    print("Vocabulary size:", len(word_to_index))
    print("Maximum length:", max_length)

    print("\nLoading caption model...")

    model = load_caption_model(
        model_file=DEFAULT_MODEL_FILE,
        vocab_size=len(word_to_index),
        max_length=max_length
    )

    print("Model loaded successfully.")

    # IMAGE TO TEST
    image_path = V1_ROOT / "data" / "images" / "1000268201_693b08cb0e.jpg"

    print("\nImage:")
    print(image_path)

    # Extract CNN feature
    print("\nExtracting image features...")

    image_feature = extract_feature(
        str(image_path),
        cnn_model
    )

    print(
        "Feature shape:",
        image_feature.shape
    )

    # Generate caption
    print("\nGenerating caption...")

    caption = generate_caption(
        model=model,
        image_feature=image_feature,
        word_to_index=word_to_index,
        index_to_word=index_to_word,
        max_length=max_length
    )

    print("\n==============================")
    print("GENERATED CAPTION")
    print("==============================")
    print(caption)
