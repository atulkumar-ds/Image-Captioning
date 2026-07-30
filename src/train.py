import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        print("GPU detected:", gpus[0])
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected — using CPU")

import os
import random

from keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

from dataset import load_captions, clean_all_captions
from vocabulary import build_vocabulary
from tokenizer import create_tokenizer
from sequences import find_max_length
from cnn_features import load_features
from data_generator import CaptionDataGenerator
from model import build_caption_model, compile_caption_model


# Configuration

FEATURE_FILE = "data/features/image_features.npz"
MODEL_DIR = "models"

MIN_FREQUENCY = 5
BATCH_SIZE = 64
EPOCHS = 20

EMBEDDING_DIM = 256
HIDDEN_DIM = 256
FEATURE_SIZE = 2048

TRAIN_RATIO = 0.90
RANDOM_SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)


# 1. Load and clean captions

print("\nLoading captions...")

captions_dict = load_captions()
captions_dict = clean_all_captions(captions_dict)

print("Total images with captions:", len(captions_dict))


# 2. Split images into train and validation

print("\nSplitting dataset...")

image_names = list(captions_dict.keys())

random.seed(RANDOM_SEED)
random.shuffle(image_names)

split_index = int(len(image_names) * TRAIN_RATIO)

train_images = image_names[:split_index]
val_images = image_names[split_index:]

train_captions = {
    image_name: captions_dict[image_name]
    for image_name in train_images
}

val_captions = {
    image_name: captions_dict[image_name]
    for image_name in val_images
}

print("Training images:", len(train_captions))
print("Validation images:", len(val_captions))


# 3. Build vocabulary
# IMPORTANT: training captions only

print("\nBuilding vocabulary...")

word_counts = build_vocabulary(train_captions)

word_to_index, index_to_word = create_tokenizer(
    word_counts,
    min_frequency=MIN_FREQUENCY
)

vocab_size = len(word_to_index)

print("Vocabulary size:", vocab_size)


# 4. Find maximum caption length

max_length = find_max_length(train_captions)

print("Maximum caption length:", max_length)


# 5. Load precomputed CNN features
#

print("\nLoading CNN features...")

features_dict = load_features(FEATURE_FILE)

print("CNN features loaded:", len(features_dict))


# 6. Create training generator

print("\nCreating training generator...")

train_generator = CaptionDataGenerator(
    captions_dict=train_captions,
    features_dict=features_dict,
    word_to_index=word_to_index,
    max_length=max_length,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# 7. Create validation generator
# 

print("Creating validation generator...")

val_generator = CaptionDataGenerator(
    captions_dict=val_captions,
    features_dict=features_dict,
    word_to_index=word_to_index,
    max_length=max_length,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nTraining batches:", len(train_generator))
print("Validation batches:", len(val_generator))


# 8. Build model

print("\nBuilding model...")

model = build_caption_model(
    vocab_size=vocab_size,
    max_length=max_length,
    feature_size=FEATURE_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM
)

model = compile_caption_model(model)

model.summary()


# 9. Callbacks

checkpoint = ModelCheckpoint(
    filepath=os.path.join(
        MODEL_DIR,
        "best_caption_model.keras"
    ),
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    mode="min",
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    mode="min",
    verbose=1
)


# 10. Train

print("\n========================================")
print("Starting Model Training")
print("========================================\n")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# 11. Save final model

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "final_caption_model.keras"
)

model.save(FINAL_MODEL_PATH)

print("\n========================================")
print("Training Complete")
print("========================================")

print("\nBest model:")
print(os.path.join(MODEL_DIR, "best_caption_model.keras"))

print("\nFinal model:")
print(FINAL_MODEL_PATH)