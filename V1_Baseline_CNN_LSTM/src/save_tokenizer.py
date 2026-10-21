import json
import os
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
V1_ROOT = SRC_DIR.parent
DEFAULT_TOKENIZER_FILE = V1_ROOT / "models" / "tokenizer.json"


def save_tokenizer(
    word_to_index,
    index_to_word,
    max_length,
    save_path=None
):
    if save_path is None:
        save_path = DEFAULT_TOKENIZER_FILE

    save_path = Path(save_path).resolve()
    os.makedirs(save_path.parent, exist_ok=True)

    tokenizer_data = {
        "word_to_index": word_to_index,
        "index_to_word": {
            str(index): word
            for index, word in index_to_word.items()
        },
        "max_length": max_length,
        "vocab_size": len(word_to_index)
    }

    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(
            tokenizer_data,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("\nTokenizer saved successfully!")
    print("Vocabulary size:", len(word_to_index))
    print("Maximum length:", max_length)
    print("Saved to:", save_path)


if __name__ == "__main__":
    import random
    from dataset import load_captions, clean_all_captions
    from vocabulary import build_vocabulary
    from tokenizer import create_tokenizer
    from sequences import find_max_length

    MIN_FREQUENCY = 5
    TRAIN_RATIO = 0.90
    RANDOM_SEED = 42

    print("\nLoading captions...")
    captions_dict = load_captions()
    captions_dict = clean_all_captions(captions_dict)

    image_names = list(captions_dict.keys())
    random.seed(RANDOM_SEED)
    random.shuffle(image_names)

    split_index = int(len(image_names) * TRAIN_RATIO)
    train_images = image_names[:split_index]

    train_captions = {
        image_name: captions_dict[image_name]
        for image_name in train_images
    }

    word_counts = build_vocabulary(train_captions)
    word_to_index, index_to_word = create_tokenizer(
        word_counts,
        min_frequency=MIN_FREQUENCY
    )

    max_length = find_max_length(train_captions)

    save_tokenizer(
        word_to_index,
        index_to_word,
        max_length,
        DEFAULT_TOKENIZER_FILE
    )
