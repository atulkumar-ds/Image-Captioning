import tensorflow as tf

from loss import loss_function

optimizer = tf.keras.optimizers.Adam(
    learning_rate=1e-4,
    clipnorm=5.0
)


@tf.function
def train_step(
    model,
    image_features,
    captions,
    hidden,
    cell
):

    total_loss = 0.0

    with tf.GradientTape() as tape:

        predictions = model(
            image_features,
            captions,
            hidden,
            cell
        )

        target = captions[:, 1:]

        sequence_length = captions.shape[1] - 1

        for t in range(sequence_length):

            total_loss += loss_function(
                target[:, t],
                predictions[:, t]
            )

        average_loss = total_loss / tf.cast(
            sequence_length,
            tf.float32
        )

    gradients = tape.gradient(
        average_loss,
        model.trainable_variables
    )

    optimizer.apply_gradients(
        zip(
            gradients,
            model.trainable_variables
        )
    )

    return average_loss