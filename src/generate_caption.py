import json
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from cnn_features import (
    cnn_model,
    extract_feature
)


# Paths

MODEL_FILE = "models/best_caption_model.keras"
TOKENIZER_FILE = "models/tokenizer.json"


# Load tokenizer

def load_tokenizer(file_path):

    with open(
        file_path,
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


# Generate caption

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


        # Predict next word

        predictions = model.predict(
            [
                image_feature.reshape(1, -1),
                sequence
            ],
            verbose=0
        )


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


# Main

if __name__ == "__main__":

    print("\nLoading caption model...")

    model = load_model(
        MODEL_FILE
    )

    print("Model loaded successfully.")


    print("\nLoading tokenizer...")

    (
        word_to_index,
        index_to_word,
        max_length
    ) = load_tokenizer(
        TOKENIZER_FILE
    )

    print("Tokenizer loaded successfully.")

    print("Vocabulary size:", len(word_to_index))
    print("Maximum length:", max_length)


    # IMAGE TO TEST

    image_path = "data/images/1000268201_693b08cb0e.jpg"

    print("\nImage:")
    print(image_path)


    # Extract CNN feature

    print("\nExtracting image features...")

    image_feature = extract_feature(
        image_path,
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