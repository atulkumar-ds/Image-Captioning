import os

from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from dataset import load_captions, clean_all_captions
from cnn_features import load_features
from generate_caption import load_tokenizer, generate_caption


MODEL_FILE = "models/best_caption_model.keras"
TOKENIZER_FILE = "models/tokenizer.json"
FEATURE_FILE = "data/features/image_features.npz"


print("Loading model...")

model = load_model(MODEL_FILE)

print("Model loaded successfully.")


word_to_index, index_to_word, max_length = load_tokenizer(
    TOKENIZER_FILE
)

print("Tokenizer loaded.")


captions_dict = load_captions()
captions_dict = clean_all_captions(captions_dict)

print("Captions loaded.")


print("Loading image features...")

features_dict = load_features(FEATURE_FILE)

print(f"Loaded {len(features_dict)} image features.")


actual = []
predicted = []

total_images = len(features_dict)

print(f"\nEvaluating {total_images} images...\n")


for i, (image_name, captions) in enumerate(captions_dict.items()):

    if image_name not in features_dict:
        continue

    image_feature = features_dict[image_name]

    prediction = generate_caption(
        model=model,
        image_feature=image_feature,
        word_to_index=word_to_index,
        index_to_word=index_to_word,
        max_length=max_length
    )

    predicted.append(prediction.split())

    references = []

    for caption in captions:

        caption = caption.replace("<start>", "")
        caption = caption.replace("<end>", "")
        caption = caption.strip()

        references.append(caption.split())

    actual.append(references)

    if (i + 1) % 500 == 0:
        print(f"Processed {i + 1}/{total_images}")


print("\nCalculating BLEU Scores...\n")

smooth = SmoothingFunction().method1

bleu1 = corpus_bleu(
    actual,
    predicted,
    weights=(1.0, 0.0, 0.0, 0.0),
    smoothing_function=smooth
)

bleu2 = corpus_bleu(
    actual,
    predicted,
    weights=(0.5, 0.5, 0.0, 0.0),
    smoothing_function=smooth
)

bleu3 = corpus_bleu(
    actual,
    predicted,
    weights=(1/3, 1/3, 1/3, 0.0),
    smoothing_function=smooth
)

bleu4 = corpus_bleu(
    actual,
    predicted,
    weights=(0.25, 0.25, 0.25, 0.25),
    smoothing_function=smooth
)


print("\n==============================")
print("BLEU SCORE RESULTS")
print("==============================")

print(f"BLEU-1 : {bleu1:.4f}")
print(f"BLEU-2 : {bleu2:.4f}")
print(f"BLEU-3 : {bleu3:.4f}")
print(f"BLEU-4 : {bleu4:.4f}")

print("==============================")