import math
import tensorflow as tf


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True,
    reduction="none"
)


def compute_loss_components(real, pred):
    """
    Computes the masked cross-entropy sum and valid (non-padding) token count.

    Padding tokens (ID 0) contribute exactly zero to the loss sum and zero to the valid token count.

    Returns:
        masked_loss_sum (tf.Tensor): Scalar sum of cross-entropy over non-padding tokens.
        valid_token_count (tf.Tensor): Scalar count of non-padding tokens.
    """
    # Create mask: True for non-padding tokens, False for padding (real == 0)
    mask = tf.math.logical_not(
        tf.math.equal(real, 0)
    )

    # Compute unreduced per-token cross entropy
    loss_ = loss_object(
        real,
        pred
    )

    mask = tf.cast(
        mask,
        dtype=loss_.dtype
    )

    # Zero out loss at padding positions
    loss_ *= mask

    masked_loss_sum = tf.reduce_sum(loss_)
    valid_token_count = tf.reduce_sum(mask)

    return masked_loss_sum, valid_token_count


def loss_function(real, pred):
    """
    Computes Sparse Categorical Crossentropy masked by non-padding tokens (real != 0),
    normalized strictly by the count of valid tokens in the batch:
        masked_loss_sum / valid_token_count

    Padding tokens (ID 0) contribute exactly zero loss and zero gradient.
    If valid_token_count is 0, safely returns 0.0 without NaN or Inf.
    Compatible with both per-step (1D real, 2D pred) and full-sequence (2D real, 3D pred) shapes.
    """
    masked_loss_sum, valid_token_count = compute_loss_components(real, pred)

    # Safe normalization: avoids division by zero if batch has 0 valid tokens
    return tf.math.divide_no_nan(masked_loss_sum, valid_token_count)


def aggregate_validation_loss(total_loss_sum, total_token_count, num_batches=1):
    """
    Computes dataset-level globally token-weighted validation loss:
        total_loss_sum / total_token_count

    Validates that:
    1. The validation dataset is not empty (num_batches > 0).
    2. The total valid target token count across the epoch is strictly positive (total_token_count > 0).
    3. The computed validation loss is finite (neither NaN nor infinite).

    Raises:
        ValueError: If num_batches <= 0 or total_token_count <= 0.
        FloatingPointError: If the computed loss is non-finite (NaN or Inf).

    Returns:
        float: Finite, globally token-weighted validation loss.
    """
    if num_batches <= 0:
        raise ValueError("Validation dataset is empty: 0 batches evaluated.")
    if total_token_count <= 0:
        raise ValueError(
            f"Validation dataset contains zero valid target tokens across {num_batches} batch(es). "
            "Cannot evaluate validation loss or update best checkpoint."
        )

    val_loss = float(total_loss_sum / total_token_count)
    if not math.isfinite(val_loss):
        raise FloatingPointError(f"Computed validation loss is non-finite ({val_loss}).")

    return val_loss


if __name__ == "__main__":
    # Hand-computed test
    y_true = tf.constant([1, 2, 0, 4])
    y_pred = tf.random.normal((4, 5000))

    loss = loss_function(
        y_true,
        y_pred
    )

    print("Loss:", float(loss))
