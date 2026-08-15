import tensorflow as tf
import numpy as np
from keras.preprocessing.sequence import pad_sequences


AUTOTUNE = tf.data.AUTOTUNE

def load_features(feature_file):

    features = np.load(
        feature_file,
        allow_pickle=True
    )

    return features.item()

def encode_caption(
    caption,
    word_to_index
):

    sequence = []

    for word in caption.split():

        sequence.append(
            word_to_index.get(
                word,
                word_to_index["<unk>"]
            )
        )

    return sequence

def create_training_samples(

    captions_dict,
    features_dict,
    word_to_index

):

    image_features = []

    input_sequences = []

    target_words = []

    for image_name, captions in captions_dict.items():

        if image_name not in features_dict:
            continue

        feature = features_dict[image_name]

        for caption in captions:

            sequence = encode_caption(
                caption,
                word_to_index
            )

            for i in range(1, len(sequence)):

                image_features.append(feature)

                input_sequences.append(
                    sequence[:i]
                )

                target_words.append(
                    sequence[i]
                )

    return (

        image_features,

        input_sequences,

        target_words

    )

def pad_caption_sequences(
    sequences,
    max_length
):

    return pad_sequences(
        sequences,
        maxlen=max_length,
        padding="post"
    )

def create_dataset(

    image_features,
    input_sequences,
    target_words,

    batch_size

):

    dataset = tf.data.Dataset.from_tensor_slices(

        (

            np.array(
                image_features,
                dtype=np.float32
            ),

            np.array(
                input_sequences,
                dtype=np.int32
            ),

            np.array(
                target_words,
                dtype=np.int32
            )

        )

    )

    dataset = dataset.shuffle(
        10000
    )

    dataset = dataset.batch(
        batch_size
    )

    dataset = dataset.prefetch(
        AUTOTUNE
    )

    return dataset

def build_training_dataset(

    captions_dict,

    features_dict,

    word_to_index,

    max_length,

    batch_size

):

    image_features, sequences, targets = create_training_samples(

        captions_dict,

        features_dict,

        word_to_index

    )

    sequences = pad_caption_sequences(

        sequences,

        max_length

    )

    dataset = create_dataset(

        image_features,

        sequences,

        targets,

        batch_size

    )

    return dataset