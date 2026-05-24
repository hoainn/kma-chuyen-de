"""
Variational Autoencoder for syscall anomaly detection.

Architecture (paper): input → hidden_dim(32) → latent(8) → hidden_dim(32) → reconstruction
Activation: SELU + dropout 0.2, L2 regularisation on Dense layers.
Anomaly score: mean squared reconstruction error per sample.
"""

from __future__ import annotations

import numpy as np


def build_vae(
    input_dim: int = 149,
    latent_dim: int = 8,
    hidden_dim: int = 32,
    dropout: float = 0.2,
    l2: float = 1e-4,
):
    """Returns (encoder, decoder, vae) Keras models — Keras 3 / TF 2.16+ compatible."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    reg = keras.regularizers.l2(l2)

    class Sampling(layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            z_lv = tf.clip_by_value(z_log_var, -10.0, 10.0)
            return z_mean + tf.exp(0.5 * z_lv) * tf.random.normal(tf.shape(z_mean))

    class KLLoss(layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            z_lv = tf.clip_by_value(z_log_var, -10.0, 10.0)
            kl = -0.5 * tf.reduce_mean(1 + z_lv - tf.square(z_mean) - tf.exp(z_lv))
            self.add_loss(kl)
            return inputs

    # Encoder: input → hidden_dim → latent
    x_in = keras.Input(shape=(input_dim,), name="encoder_input")
    x = layers.Dense(hidden_dim, activation="selu", kernel_regularizer=reg)(x_in)
    x = layers.Dropout(dropout)(x)
    z_mean    = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling(name="z")([z_mean, z_log_var])
    [z_mean, z_log_var] = KLLoss(name="kl_loss")([z_mean, z_log_var])
    encoder = keras.Model(x_in, [z_mean, z_log_var, z], name="encoder")

    # Decoder: latent → hidden_dim → input
    z_in = keras.Input(shape=(latent_dim,), name="decoder_input")
    x = layers.Dense(hidden_dim, activation="selu", kernel_regularizer=reg)(z_in)
    x = layers.Dropout(dropout)(x)
    x_out = layers.Dense(input_dim, name="decoder_output")(x)
    decoder = keras.Model(z_in, x_out, name="decoder")

    vae = keras.Model(x_in, decoder(encoder(x_in)[2]), name="vae")
    return encoder, decoder, vae
