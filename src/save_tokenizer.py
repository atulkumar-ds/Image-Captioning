import json
import random

from dataset import load_captions, clean_all_captions
from vocabulary import build_vocabulary
from tokenizer import create_tokenizer
from sequences import find_max_length


# Same settings used during training

MIN_FREQUENCY = 5
TRAIN_RATIO = 0.90
RANDOM_SEED = 42

TOKENIZER_FILE = "models/tokenizer.json"


# 1. Load captions

print("\nLoading captions...")

captions_dict = load_captions()
captions_dict = clean_all_captions(captions_dict)


# 2. Reproduce EXACT training split

image_names = list(captions_dict.keys())

random.seed(RANDOM_SEED)
random.shuffle(image_names)

split_index = int(
    len(image_names) * TRAIN_RATIO
)

train_images = image_names[:split_index]

train_captions = {
    image_name: captions_dict[image_name]
    for image_name in train_images
}


# 3. Rebuild EXACT training vocabulary

word_counts = build_vocabulary(
    train_captions
)

word_to_index, index_to_word = create_tokenizer(
    word_counts,
    min_frequency=MIN_FREQUENCY
)


# 4. Maximum length

max_length = find_max_length(
    train_captions
)


# 5. Save tokenizer information

tokenizer_data = {

    "word_to_index": word_to_index,

    "index_to_word": {
        str(index): word
        for index, word in index_to_word.items()
    },

    "max_length": max_length,

    "vocab_size": len(word_to_index)
}


with open(
    TOKENIZER_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        tokenizer_data,
        file,
        ensure_ascii=False,
        indent=4
    )


print("\nTokenizer saved successfully!")

print("Vocabulary size:", len(word_to_index))
print("Maximum length:", max_length)
print("Saved to:", TOKENIZER_FILE)