import tensorflow as tf
from keras.layers import Dense


class CNNEncoder(tf.keras.Model):

    def __init__(self, embedding_dim):

        super().__init__()

        self.fc = Dense(
            embedding_dim,
            activation="relu"
        )

    def call(self, features):

        features = self.fc(features)

        return features

if __name__ == "__main__":

    encoder = CNNEncoder(
        embedding_dim=256
    )

    dummy_features = tf.random.normal(
        (1, 64, 2048)
    )

    output = encoder(dummy_features)

    print("Input Shape :", dummy_features.shape)
    print("Output Shape:", output.shape)