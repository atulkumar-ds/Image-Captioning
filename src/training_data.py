from keras.preprocessing.sequence import pad_sequences
from dataset import load_captions, clean_all_captions
from vocabulary import build_vocabulary
from tokenizer import create_tokenizer
from sequences import find_max_length
from cnn_features import load_features


def create_caption_samples(sequence, max_length):

    input_sequences = []
    target_words = []

    for i in range(1, len(sequence)):

        input_sequence = sequence[:i]

        target_word = sequence[i]

        input_sequence = pad_sequences(
            [input_sequence],
            maxlen=max_length,
            padding="post"
        )[0]

        input_sequences.append(input_sequence)
        target_words.append(target_word)

    return input_sequences, target_words


if __name__ == "__main__":

    sequence = [1, 4, 5, 6, 7, 2]

    input_sequences, target_words = create_caption_samples(
        sequence,
        max_length=10
    )

    for x, y in zip(input_sequences, target_words):
        print("Input :", x)
        print("Target:", y)
        print()



if __name__ == "__main__":

    # 1. Load and clean captions

    captions_dict = load_captions()
    captions_dict = clean_all_captions(captions_dict)

    print("Total caption images:", len(captions_dict))


    # 2. Build vocabulary

    word_counts = build_vocabulary(captions_dict)

    word_to_index, index_to_word = create_tokenizer(
        word_counts,
        min_frequency=5
    )

    print("Vocabulary size:", len(word_to_index))


    # 3. Find maximum caption length

    max_length = find_max_length(captions_dict)

    print("Maximum caption length:", max_length)


    # 4. Load CNN image features

    features_dict = load_features(
        "data/features/image_features.npz"
    )

    print("Total image features:", len(features_dict))


    # 5. Pick ONE real image

    image_name = list(captions_dict.keys())[0]

    print("\nImage:")
    print(image_name)

    image_feature = features_dict[image_name]

    print("\nImage feature shape:")
    print(image_feature.shape)


    # 6. Pick ONE real caption

    caption = captions_dict[image_name][0]

    print("\nCaption:")
    print(caption)


    # 7. Encode caption

    sequence = []

    for word in caption.split():

        index = word_to_index.get(
            word,
            word_to_index["<unk>"]
        )

        sequence.append(index)

    print("\nEncoded caption:")
    print(sequence)


    # 8. Create training samples

    input_sequences, target_words = create_caption_samples(
        sequence,
        max_length
    )


    # 9. Display training samples

    print("\nTraining Samples:")

    for input_sequence, target_word in zip(
        input_sequences,
        target_words
    ):

        # Remove padding just for readable output
        input_words = [
            index_to_word[index]
            for index in input_sequence
            if index != 0
        ]

        target = index_to_word[target_word]

        print(
            " ".join(input_words),
            " -> ",
            target
        )