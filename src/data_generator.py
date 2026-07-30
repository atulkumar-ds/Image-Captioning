import numpy as np
from keras.utils import Sequence


class CaptionDataGenerator(Sequence):

    def __init__(
        self,
        captions_dict,
        features_dict,
        word_to_index,
        max_length,
        batch_size=64,
        shuffle=True
    ):

        self.captions_dict = captions_dict
        self.features_dict = features_dict
        self.word_to_index = word_to_index
        self.max_length = max_length
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.samples = self._build_sample_index()

        self.indices = np.arange(len(self.samples))

        self.on_epoch_end()


    def _build_sample_index(self):

        samples = []

        for image_name, captions in self.captions_dict.items():

            if image_name not in self.features_dict:
                continue

            for caption in captions:

                sequence = [
                    self.word_to_index.get(
                        word,
                        self.word_to_index["<unk>"]
                    )
                    for word in caption.split()
                ]

                for i in range(1, len(sequence)):

                    samples.append(
                        (image_name, sequence, i)
                    )

        return samples


    def __len__(self):

        return int(
            np.ceil(
                len(self.samples) / self.batch_size
            )
        )


    def __getitem__(self, batch_index):

        start = batch_index * self.batch_size

        end = min(
            start + self.batch_size,
            len(self.samples)
        )

        batch_indices = self.indices[start:end]

        image_batch = []
        sequence_batch = []
        target_batch = []

        for index in batch_indices:

            image_name, sequence, position = self.samples[index]

            image_feature = self.features_dict[image_name]

            input_sequence = sequence[:position]

            target_word = sequence[position]

            padded_sequence = np.zeros(
                self.max_length,
                dtype=np.int32
            )

            length = min(
                len(input_sequence),
                self.max_length
            )

            padded_sequence[:length] = input_sequence[:length]

            image_batch.append(image_feature)
            sequence_batch.append(padded_sequence)
            target_batch.append(target_word)


        return (
            (
                np.array(image_batch, dtype=np.float32),
                np.array(sequence_batch, dtype=np.int32)
            ),
            np.array(target_batch, dtype=np.int32)
        )


    def on_epoch_end(self):

        if self.shuffle:
            np.random.shuffle(self.indices)

if __name__ == "__main__":

    from dataset import load_captions, clean_all_captions
    from vocabulary import build_vocabulary
    from tokenizer import create_tokenizer
    from sequences import find_max_length
    from cnn_features import load_features

    # 1. Captions
    captions_dict = load_captions()
    captions_dict = clean_all_captions(captions_dict)

    # 2. Vocabulary + tokenizer
    word_counts = build_vocabulary(captions_dict)

    word_to_index, index_to_word = create_tokenizer(
        word_counts,
        min_frequency=5
    )

    # 3. Maximum caption length
    max_length = find_max_length(captions_dict)

    # 4. CNN features
    features_dict = load_features(
        "data/features/image_features.npz"
    )

    # 5. Create generator
    generator = CaptionDataGenerator(
        captions_dict=captions_dict,
        features_dict=features_dict,
        word_to_index=word_to_index,
        max_length=max_length,
        batch_size=64,
        shuffle=True
    )

    print("\nTotal training samples:", len(generator.samples))
    print("Total batches:", len(generator))

    # 6. Get first batch
    (image_batch, sequence_batch), target_batch = generator[0]

    print("\nImage batch shape:")
    print(image_batch.shape)

    print("\nSequence batch shape:")
    print(sequence_batch.shape)

    print("\nTarget batch shape:")
    print(target_batch.shape)

    print("\nExample:")
    print("Image feature shape:", image_batch[0].shape)
    print("Caption sequence:", sequence_batch[0])
    print("Target index:", target_batch[0])
    print("Target word:", index_to_word[target_batch[0]])