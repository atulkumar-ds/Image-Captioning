import tensorflow as tf

from encoder import CNNEncoder
from decoder import RNNDecoder


class ImageCaptionModel(tf.keras.Model):

    def __init__(
        self,
        embedding_dim,
        units,
        vocab_size
    ):

        super().__init__()

        self.encoder = CNNEncoder(
            embedding_dim
        )

        self.decoder = RNNDecoder(
            embedding_dim,
            units,
            vocab_size
        )

    def call(
        self,
        features,
        caption,
        hidden,
        cell
    ):

        # Encode image features
        features = self.encoder(features)

        outputs = []

        attention_maps = []

        # Teacher Forcing
        for t in range(caption.shape[1] - 1):

            current_word = caption[:, t:t+1]

            prediction, hidden, cell, attention = self.decoder(
                current_word,
                features,
                hidden,
                cell
            )

            outputs.append(prediction)

            attention_maps.append(attention)

        outputs = tf.stack(
            outputs,
            axis=1
        )

        return outputs

if __name__ == "__main__":

    batch = 2
    vocab_size = 5000
    embedding_dim = 256
    units = 256
    max_length = 20

    model = ImageCaptionModel(
        embedding_dim,
        units,
        vocab_size
    )

    features = tf.random.normal(
        (batch,64,2048)
    )

    captions = tf.ones(
        (batch,max_length),
        dtype=tf.int32
    )

    hidden = tf.zeros(
        (batch,units)
    )

    cell = tf.zeros(
        (batch,units)
    )

    output = model(
        features,
        captions,
        hidden,
        cell
    )

    print(output.shape)