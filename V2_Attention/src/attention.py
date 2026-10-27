import tensorflow as tf
from keras.layers import Dense


class BahdanauAttention(tf.keras.layers.Layer):

    def __init__(self, units):

        super().__init__()

        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, features, hidden):

        # hidden:
        # (batch_size, hidden_dim)

        hidden = tf.expand_dims(hidden, axis=1)

        # score:
        # (batch_size, 64, 1)

        score = self.V(
            tf.nn.tanh(
                self.W1(features) +
                self.W2(hidden)
            )
        )

        # attention weights
        attention_weights = tf.nn.softmax(
            score,
            axis=1
        )

        # context vector
        context_vector = attention_weights * features

        context_vector = tf.reduce_sum(
            context_vector,
            axis=1
        )

        return context_vector, attention_weights

if __name__ == "__main__":

    attention = BahdanauAttention(256)

    features = tf.random.normal(
        (2, 64, 256)
    )

    hidden = tf.random.normal(
        (2, 256)
    )

    context_vector, attention_weights = attention(
        features,
        hidden
    )

    print("Features Shape :", features.shape)
    print("Hidden Shape   :", hidden.shape)
    print("Context Shape  :", context_vector.shape)
    print("Weights Shape  :", attention_weights.shape)