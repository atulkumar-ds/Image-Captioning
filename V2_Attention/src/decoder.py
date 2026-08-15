import tensorflow as tf

from keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Dropout
)

from attention import BahdanauAttention


class RNNDecoder(tf.keras.Model):

    def __init__(
        self,
        embedding_dim,
        units,
        vocab_size
    ):

        super().__init__()

        self.units = units

        self.embedding = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim
        )

        self.attention = BahdanauAttention(
            units
        )

        self.lstm = LSTM(
            units,
            return_sequences=True,
            return_state=True
        )

        self.fc1 = Dense(
            units,
            activation="relu"
        )

        self.dropout = Dropout(0.3)

        self.fc2 = Dense(
            vocab_size
        )

    def call(
        self,
        x,
        features,
        hidden,
        cell
    ):

        context_vector, attention_weights = self.attention(
            features,
            hidden
        )

        x = self.embedding(x)

        context_vector = tf.expand_dims(
            context_vector,
            axis=1
        )

        x = tf.concat(
            [context_vector, x],
            axis=-1
        )

        output, hidden, cell = self.lstm(
            x,
            initial_state=[hidden, cell]
        )

        output = tf.reshape(
            output,
            (-1, output.shape[2])
        )

        output = self.fc1(output)

        output = self.dropout(output)

        output = self.fc2(output)

        return (
            output,
            hidden,
            cell,
            attention_weights
        )

    def reset_state(
        self,
        batch_size
    ):

        hidden = tf.zeros(
            (batch_size, self.units)
        )

        cell = tf.zeros(
            (batch_size, self.units)
        )

        return hidden, cell

if __name__ == "__main__":

    vocab_size = 5000
    embedding_dim = 256
    units = 256

    decoder = RNNDecoder(
        embedding_dim,
        units,
        vocab_size
    )

    features = tf.random.normal(
        (2, 64, 256)
    )

    hidden = tf.zeros(
        (2, units)
    )

    cell = tf.zeros(
        (2, units)
    )

    words = tf.constant(
        [[1], [1]]
    )

    output, hidden, cell, attention = decoder(
        words,
        features,
        hidden,
        cell
    )

    print("Output:", output.shape)
    print("Hidden:", hidden.shape)
    print("Cell:", cell.shape)
    print("Attention:", attention.shape)