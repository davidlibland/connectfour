import tensorflow as tf

def sample_gumbel(shape):
    uniform_samples = tf.random_uniform(
        shape,
        minval=1e-5,
        maxval=1. - 1e-5
    )
    return -tf.log(-tf.log(uniform_samples))

def sample_masked_multinomial(logits, mask, axis=None):
    gumbels = sample_gumbel(logits.shape)
    noisy_logits = logits + gumbels
    min_val = tf.broadcast_to(tf.reduce_min(noisy_logits) - 1, logits.shape)
    masked_logits = tf.where(mask, min_val, noisy_logits)
    return tf.argmax(masked_logits, axis=axis)