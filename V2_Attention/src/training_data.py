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
    shuffle=True,
    seed=None,
):

    def generator():

        for image_name, image_captions in captions_dict.items():

            if image_name not in features_dict:
                continue

            feature = np.asarray(features_dict[image_name], dtype=np.float32)
            if feature.shape != (64, 2048):
                if feature.size == 64 * 2048:
                    feature = feature.reshape(64, 2048)
                else:
                    raise ValueError(
                        f"Image feature for '{image_name}' has unexpected shape {feature.shape} "
                        f"(size {feature.size}), expected 64x2048."
                    )

            for caption in image_captions:

                sequence = encode_caption(
                    caption,
                    word_to_index
                )

                end_token_id = word_to_index.get("<end>")

                sequence = pad_sequence(
                    sequence,
                    max_length,
                    preserve_end_token=True,
                    end_token=end_token_id
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

    if shuffle:
        if seed is not None:
            dataset = dataset.shuffle(256, seed=seed, reshuffle_each_iteration=True)
        else:
            dataset = dataset.shuffle(256)

    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(1)

    return dataset
