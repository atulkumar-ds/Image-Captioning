import tensorflow as tf
import numpy as np
from tokenizer import encode_caption
from sequences import pad_sequence


def build_training_dataset(
    captions_dict,
    features_dict,
    word_to_index,
    max_length,
    batch_size=64,
):

    def generator():

        for image_name, image_captions in captions_dict.items():

            if image_name not in features_dict:
                continue

            feature = features_dict[image_name]

            for caption in image_captions:

                sequence = encode_caption(
                    caption,
                    word_to_index
                )

                sequence = pad_sequence(
                    sequence,
                    max_length
                )

                yield (
                    np.asarray(feature, dtype=np.float32),
                    np.asarray(sequence, dtype=np.int32)
                )

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(
                shape=(64, 2048),
                dtype=tf.float32
            ),
            tf.TensorSpec(
                shape=(max_length,),
                dtype=tf.int32
            )
        )
    )

    dataset = dataset.shuffle(256)

    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(1)

    return dataset