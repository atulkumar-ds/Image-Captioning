import os
import random

from keras.models import load_model

from dataset import (
    load_captions,
    clean_all_captions
)

from generate_caption import (
    load_tokenizer,
    generate_caption
)

from cnn_features import (
    cnn_model,
    extract_feature
)



MODEL_FILE = "models/best_caption_model.keras"
TOKENIZER_FILE = "models/tokenizer.json"

IMAGE_DIR = "data/images"

NUMBER_OF_IMAGES = 10



print("Loading model...\n")

model = load_model(MODEL_FILE)

print("Model loaded.")



(
    word_to_index,
    index_to_word,
    max_length
) = load_tokenizer(TOKENIZER_FILE)

print("Vocabulary size:", len(word_to_index))
print("Maximum caption length:", max_length)



print("\nLoading captions...\n")

captions_dict = load_captions()
captions_dict = clean_all_captions(captions_dict)

print("Images in dataset:", len(captions_dict))



# image_names = list(captions_dict.keys())

# random.shuffle(image_names)

# selected_images = image_names[:NUMBER_OF_IMAGES]
image_names = list(captions_dict.keys())

selected_images = [
    "416650559_cd08d3cd96.jpg"
]



print("\n")
print("=" * 80)
print("IMAGE CAPTION EVALUATION")
print("=" * 80)


for image_name in selected_images:
    
    image_path = os.path.join(
        IMAGE_DIR,
        image_name
    )

    print("\n")
    print("=" * 80)
    print("IMAGE:")
    print(image_name)

    print("\nExtracting features...")

    image_feature = extract_feature(
        image_path,
        cnn_model
    )

    predicted_caption = generate_caption(
        model=model,
        image_feature=image_feature,
        word_to_index=word_to_index,
        index_to_word=index_to_word,
        max_length=max_length
    )

    print("\nGROUND TRUTH CAPTIONS:\n")

    for caption in captions_dict[image_name]:
        print("-", caption)

    print("\nPREDICTED CAPTION:\n")
    print(predicted_caption)

    print("=" * 80)


print("\nEvaluation Complete.")