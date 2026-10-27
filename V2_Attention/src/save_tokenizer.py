import json
import os


def save_tokenizer(
    word_to_index,
    index_to_word,
    max_length,
    save_path,
):
    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )

    tokenizer_data = {

        "word_to_index": word_to_index,

        "index_to_word": {
            str(index): word
            for index, word in index_to_word.items()
        },

        "max_length": max_length,

        "vocab_size": len(word_to_index),
    }

    with open(
        save_path,
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
    print("Saved to:", save_path)