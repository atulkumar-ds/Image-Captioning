import os
import time
import random
import tensorflow as tf

from dataset import load_captions, clean_all_captions
from vocabulary import build_vocabulary
from tokenizer import create_tokenizer
from sequences import find_max_length

from cnn_features import load_features
from training_data import build_training_dataset

from model import ImageCaptionModel
from train_step import train_step
from loss import loss_function
from save_tokenizer import save_tokenizer


# ======================================================
# Configuration
# ======================================================

FEATURE_FILE = "model/v2_features.npy"
MODEL_DIR = "models"

BATCH_SIZE = 32
EPOCHS = 40

EMBEDDING_DIM = 256
UNITS = 256
MIN_FREQUENCY = 5

VAL_SPLIT = 0.10

os.makedirs(MODEL_DIR, exist_ok=True)


# ======================================================
# Validation Step
# ======================================================

@tf.function
def validation_step(
    model,
    image_features,
    captions,
    hidden,
    cell
):

    predictions = model(
        image_features,
        captions,
        hidden,
        cell
    )

    targets = captions[:, 1:]

    sequence_length = captions.shape[1] - 1

    total_loss = 0.0

    for t in range(sequence_length):

        total_loss += loss_function(
            targets[:, t],
            predictions[:, t]
        )

    average_loss = total_loss / tf.cast(
        sequence_length,
        tf.float32
    )

    return average_loss


# ======================================================
# Load Captions
# ======================================================

print("\nLoading captions...")

captions_dict = load_captions()
captions_dict = clean_all_captions(captions_dict)

print("Total Images:", len(captions_dict))


# ======================================================
# Train / Validation Split
# ======================================================

image_names = list(captions_dict.keys())

random.seed(42)
random.shuffle(image_names)

val_size = int(len(image_names) * VAL_SPLIT)

val_images = image_names[:val_size]
train_images = image_names[val_size:]

train_captions = {
    image: captions_dict[image]
    for image in train_images
}

val_captions = {
    image: captions_dict[image]
    for image in val_images
}

print("\nDataset Split")
print("Train Images:", len(train_captions))
print("Val Images  :", len(val_captions))


# ======================================================
# Build Vocabulary
# ======================================================

print("\nBuilding vocabulary...")

word_counts = build_vocabulary(
    train_captions
)

word_to_index, index_to_word = create_tokenizer(
    word_counts,
    min_frequency=MIN_FREQUENCY
)

max_length = find_max_length(
    train_captions
)

print("Vocabulary Size:", len(word_to_index))
print("Max Length     :", max_length)


# ======================================================
# Load CNN Features
# ======================================================

print("\nLoading CNN features...")

features_dict = load_features(
    FEATURE_FILE
)


# ======================================================
# Build Train Dataset
# ======================================================

print("\nBuilding training dataset...")

train_dataset = build_training_dataset(
    train_captions,
    features_dict,
    word_to_index,
    max_length,
    BATCH_SIZE
)


# ======================================================
# Build Validation Dataset
# ======================================================

print("Building validation dataset...")

val_dataset = build_training_dataset(
    val_captions,
    features_dict,
    word_to_index,
    max_length,
    BATCH_SIZE
)


# ======================================================
# Save Tokenizer
# ======================================================

save_tokenizer(
    word_to_index,
    index_to_word,
    max_length,
    os.path.join(
        MODEL_DIR,
        "tokenizer.json"
    )
)


# ======================================================
# Build Model
# ======================================================

model = ImageCaptionModel(
    EMBEDDING_DIM,
    UNITS,
    len(word_to_index)
)

print("\nModel Created Successfully!")


# ======================================================
# Training
# ======================================================

best_val_loss = float("inf")

for epoch in range(EPOCHS):

    start = time.time()

    # -------------------------------
    # Training
    # -------------------------------

    total_train_loss = 0.0
    train_batches = 0

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    for image_features, captions in train_dataset:

        batch_size = tf.shape(
            image_features
        )[0]

        hidden, cell = model.decoder.reset_state(
            batch_size
        )

        loss = train_step(
            model,
            image_features,
            captions,
            hidden,
            cell
        )

        total_train_loss += loss
        train_batches += 1

    train_loss = float(
        total_train_loss / train_batches
    )


    # -------------------------------
    # Validation
    # -------------------------------

    total_val_loss = 0.0
    val_batches = 0

    for image_features, captions in val_dataset:

        batch_size = tf.shape(
            image_features
        )[0]

        hidden, cell = model.decoder.reset_state(
            batch_size
        )

        loss = validation_step(
            model,
            image_features,
            captions,
            hidden,
            cell
        )

        total_val_loss += loss
        val_batches += 1

    val_loss = float(
        total_val_loss / val_batches
    )


    # -------------------------------
    # Print Results
    # -------------------------------

    elapsed = time.time() - start

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Val Loss  : {val_loss:.4f}"
    )

    print(
        f"Time      : {elapsed:.2f}s"
    )


    # -------------------------------
    # Save Best Model
    # -------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        model.save_weights(
            os.path.join(
                MODEL_DIR,
                "v2_attention.weights.h5"
            )
        )

        print(
            "✓ Best model saved."
        )


print("\nTraining Complete!")