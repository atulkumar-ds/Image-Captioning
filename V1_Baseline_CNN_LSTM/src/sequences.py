from dataset import load_captions, clean_all_captions
from vocabulary import build_vocabulary
from tokenizer import create_tokenizer, encode_caption

def find_max_length(caption_dict):
    max_length=0

    for image_name in caption_dict:
        for caption in caption_dict[image_name]:
            length=len(caption.split())

            if length>max_length:
                max_length=length

    return max_length

def pad_sequence(sequence, max_length):

    padding_needed = max_length - len(sequence)

    padded_sequence = sequence + [0] * padding_needed

    return padded_sequence


def create_training_sequences(sequence, max_length):

    input_sequences = []
    target_words = []

    for i in range(1, len(sequence)):

        input_sequence = sequence[:i]
        target_word = sequence[i]

        input_sequence = pad_sequence(
            input_sequence,
            max_length
        )

        input_sequences.append(input_sequence)
        target_words.append(target_word)

    return input_sequences, target_words

if __name__ == "__main__":

    caption_dict=load_captions()
    caption_dict=clean_all_captions(caption_dict)

    max_length=find_max_length(caption_dict)

    print("Maximum Caption Length:", max_length)
    word_counts = build_vocabulary(caption_dict)

    word_to_index, index_to_word = create_tokenizer(
        word_counts,
        min_frequency=5
    )

    first_image = list(caption_dict.keys())[0]
    caption = caption_dict[first_image][0]

    sequence = encode_caption(
        caption,
        word_to_index
    )

    padded_sequence = pad_sequence(
        sequence,
        max_length
    )

    print("\nCaption:")
    print(caption)

    print("\nOriginal Sequence:")
    print(sequence)

    print("Length:", len(sequence))

    print("\nPadded Sequence:")
    print(padded_sequence)

    print("Length:", len(padded_sequence))
    input_sequences, target_words = create_training_sequences(
    sequence,
    max_length
    )

    print("\nNumber of Training Examples:", len(input_sequences))

    print("\nFirst 5 Training Examples:")

    for i in range(5):

        print("\nInput:")
        print(input_sequences[i])

        print("Target:")
        print(target_words[i])