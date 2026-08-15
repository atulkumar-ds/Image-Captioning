from dataset import load_captions, clean_all_captions
from vocabulary import build_vocabulary


def create_tokenizer(word_count, min_frequency=5):

    word_to_index={
        "<pad>":0,
        "<start>":1,
        "<end>":2,
        "<unk>":3
        
    }
    index_to_word={
        0:"<pad>",
        1:"<start>",
        2:"<end>",
        3:"<unk>"
        
    }
    index = 4
    for word, frequency in word_count.items():
        if frequency>=min_frequency:
            if word not in word_to_index:

                word_to_index[word] = index
                index_to_word[index] = word

                index+=1
                
    return word_to_index, index_to_word


def encode_caption(caption, word_to_index):

    words = caption.split()

    sequence = []

    for word in words:

        index = word_to_index.get(
            word,
            word_to_index["<unk>"]
        )

        sequence.append(index)

    return sequence

def decode_sequence(sequence, index_to_word):

    words = []

    for index in sequence:

        word = index_to_word.get(index, "<unk>")

        words.append(word)

    caption = " ".join(words)

    return caption


if __name__ == "__main__":

    captions_dict = load_captions()
    captions_dict = clean_all_captions(captions_dict)

    word_counts = build_vocabulary(captions_dict)

    word_to_index, index_to_word = create_tokenizer(
        word_counts,
        min_frequency=5
    )

    print("Vocabulary Size:", len(word_to_index))

    print("\nFirst 20 Word → Index mappings:")

    for word, index in list(word_to_index.items())[:20]:
        print(word, "->", index)

    first_image = list(captions_dict.keys())[0]

    caption = captions_dict[first_image][0]

    sequence = encode_caption(
        caption,
        word_to_index
    )

    print("\nOriginal Caption:")
    print(caption)
    # encode
    print("\nEncoded Caption:")
    print(sequence)
    decoded_caption = decode_sequence(
    sequence,
    index_to_word
    )
    # decode
    print("\nDecoded Caption:")
    print(decoded_caption)