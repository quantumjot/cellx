from typing import Optional

import tensorflow as tf
from tensorflow import keras as K


class RandomNormalSampler(K.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z."""

    def call(self, x, training: Optional[bool] = None):
        z_mean, z_log_var = x
        epsilon = K.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAESampler(K.layers.Layer):
    """VAE Sampler.

    Parameters
    ----------
    latent_dims : int
        The number of latent dimensions to sample.
    intermediate_dims : int, Optional
        The size of the intermediate representation.  The intermediate FC
        connected layer sits between the convolutional encoder and the
        sampling layer. By specifying the intermediate layer, an additional
        FC layer is added before the sampler.
    """

    def __init__(
        self, latent_dims: int = 32, intermediate_dims: Optional[int] = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.sampler = RandomNormalSampler()
        self.z_mean = K.layers.Dense(latent_dims, name="z_mean",)
        self.z_log_var = K.layers.Dense(
            latent_dims, activation="softplus", name="z_log_var",
        )

        if intermediate_dims is not None:
            self._fc_1 = K.layers.Dense(
                intermediate_dims, activation="swish", name="intermediate_FC1",
            )
            self._fc_2 = K.layers.Dense(
                intermediate_dims, activation="swish", name="intermediate_FC2",
            )
        else:
            self._fc1 = K.layers.Lambda(lambda x: x)
            self._fc2 = K.layers.Lambda(lambda x: x)

    def call(self, x, training: Optional[bool] = None):
        x = K.layers.Flatten()(x)
        x_fc1 = self._fc1(x)
        x_fc2 = self._fc2(x)
        z_mean = self.z_mean(x_fc1)
        z_log_var = self.z_log_var(x_fc2)
        z = self.sampler([z_mean, z_log_var], training=training)
        return z_mean, z_log_var, z


if __name__ == "__main__":
    pass
