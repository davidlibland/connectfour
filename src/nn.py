import tensorflow as tf
import math


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


def gaussian_neg_log_likelihood(mu, log_sig, x):
    """1-dim gaussian"""
    l2_diff = (mu-x)**2
    scaled_l2 = l2_diff*tf.exp(-log_sig*2)/2
    log_z = -log_sig - tf.log(2*math.pi)/2
    return scaled_l2 - log_z
