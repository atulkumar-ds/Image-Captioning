import tensorflow as tf


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True,
    reduction="none"
)


def loss_function(real, pred):

    mask = tf.math.logical_not(
        tf.math.equal(real, 0)
    )

    loss = loss_object(
        real,
        pred
    )

    mask = tf.cast(
        mask,
        dtype=loss.dtype
    )

    loss *= mask

    return tf.reduce_mean(loss)

if __name__ == "__main__":

    y_true = tf.constant(
        [1, 2, 0, 4]
    )

    y_pred = tf.random.normal(
        (4, 5000)
    )

    loss = loss_function(
        y_true,
        y_pred
    )

    print(loss)