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

    with tf.GradientTape() as tape:

        predictions = model(
            image_features,
            captions,
            hidden,
            cell
        )

        target = captions[:, 1:]

        # Token-normalized masked loss across entire batch
        loss = loss_function(
            target,
            predictions
        )

    gradients = tape.gradient(
        loss,
        model.trainable_variables
    )

    optimizer.apply_gradients(
        zip(
            gradients,
            model.trainable_variables
        )
    )

    return loss
