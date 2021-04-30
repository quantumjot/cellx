from typing import Optional

import tensorflow as tf
from tensorflow import keras as K


class RandomNormalSampler(K.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z."""

    def call(self, x, training: Optional[bool] = None):
        z_mean, z_log_var = x
        epsilon = K.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


if __name__ == "__main__":
    pass
