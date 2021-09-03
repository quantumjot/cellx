from typing import Callable, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras as K

from ..layers import VAESampler


class VAEReshapeLatents(K.layers.Layer):
    """VAE latent reshaper.

    Parameters
    ----------
    shape : tuple
        The 2D shape of the reshaped latent vector.
    latent_dims : int
        The number of latent dimensions to sample.
    """

    def __init__(self, shape: Tuple[int], latent_dims: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.dense = K.layers.Dense(
            np.prod(shape), activation="swish", name="reshape_FC",
        )
        self.reshape = K.layers.Reshape(shape, name="reshape")

    def call(self, x, training: Optional[bool] = None):
        x = self.dense(x)
        x = self.reshape(x)
        return x


def mse_loss(x, reconstruction):
    """Mean Squared Error (MSE) loss for VAE reconstruction.

    Parameters
    ----------
    x : tensor
        The tensor representing the ground truth image.
    reconstruction : tensor
        The tensor representing the reconstructed image.

    Returns
    -------
    reconstruction_loss : tensor
        The reconstruction loss according to the metric.
    """
    reconstruction_loss = tf.reduce_mean(
        tf.reduce_sum(tf.math.squared_difference(x, reconstruction), axis=[1, 2, 3])
    )
    return reconstruction_loss


def ssim_loss(x, reconstruction):
    """Structural Similarity Index Measure (SSIM) loss for VAE reconstruction.

    Parameters
    ----------
    x : tensor
        The tensor representing the ground truth image.
    reconstruction : tensor
        The tensor representing the reconstructed image.

    Returns
    -------
    reconstruction_loss : tensor
        The reconstruction loss according to the metric.
    """
    # x_scaled = tf.clip_by_value((x + 4.0) * 255.0, 0, 255)
    # r_scaled = tf.clip_by_value((reconstruction + 4.0) * 255.0, 0, 255)
    power_factors = [pf * (1 / 0.6305) for pf in (0.0448, 0.2856, 0.3001)]

    reconstruction_loss = tf.image.ssim_multiscale(
        x,
        reconstruction,
        255,
        filter_size=11,
        filter_sigma=1.5,
        k1=0.01,
        k2=0.03,
        power_factors=power_factors,
    )

    return reconstruction_loss


class VAECapacity(K.Model):
    """β-Variational AutoEncoder with Capacity loss.

    β-VAE with variable capacity during training.

    Parameters
    ----------
    encoder : K.layers.Layer
        A convolutional encoder layer or model.
    decoder : K.layers.Layer
        A convolutional decoder layer or model.
    sampler : K.layers.Layer
        A sampler layer to sample.
    input_shape : tuple (W, H, C)
        The shape of the input image.
    latent_dims : int
        The size of the latent space representation.
    intermediate_dims : int, Optional
        The size of the intermediate representation.  The intermediate FC
        connected layer sits between the convolutional encoder and the
        sampling layer. By specifying the intermediate layer, an additional
        FC layer is added before the sampler.
    capacity : float
        The capacity of the model.
    gamma : float
        The loss scaling factor.
    max_iterations : int
        The number of iterations over which to linearly increase the capacity.
    reconstruction_loss_fn : Callable
        A function for calculating the reconstruction loss of the output.

    References
    ----------
    β-VAE: Learning basic visual concepts with a constrained variational framework
    Higgins, Matthey, Pal, Burgess, Glorot, Botvinick, Mohamed and Lerchner
    ICLR 2017

    Understanding disentangling in β-VAE
    Burgess, Higgins, Pal, Matthey, Watters, Desjardins and Lerchner
    https://arxiv.org/pdf/1804.03599.pdf
    """

    def __init__(
        self,
        encoder: K.layers.Layer,
        decoder: K.layers.Layer,
        sampler: K.layers.Layer = VAESampler,
        input_shape: tuple = (64, 64, 2),
        latent_dims: int = 32,
        intermediate_dims: Optional[int] = None,
        capacity: float = 50.0,
        gamma: float = 1e5,
        max_iterations: int = 10_000,
        reconstruction_loss_fn: Optional[Callable] = mse_loss,
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = sampler(
            latent_dims=latent_dims, intermediate_dims=intermediate_dims
        )
        self.gamma = gamma
        self.capacity = capacity
        self.max_iter = max_iterations
        self._iter = tf.Variable(0.0, trainable=False)
        self._reconstruction_loss_fn = reconstruction_loss_fn

        downsamples = len(encoder.layers) - 1
        bottleneck_shape = [d // 2 ** downsamples for d in input_shape[0:2]]
        shape = tuple(bottleneck_shape + [decoder.layers[0].conv.filters])
        self.reshape = VAEReshapeLatents(shape=shape, latent_dims=latent_dims)

        self._decoder_output = K.layers.Conv2D(
            filters=input_shape[-1], kernel_size=1, activation="linear", name="output"
        )

        # self._mse = K.losses.MeanSquaredError()
        self._n_pixels = np.prod(input_shape)

    def train_step(self, x):
        """Training step for VAE."""

        if isinstance(x, tuple):
            x = x[0]

        with tf.GradientTape() as tape:
            # forward pass
            z_mean, z_log_var, z = self.encode(x)
            reconstruction = self.decode(z)

            # set the capacity as a function of training iteration
            capacity = (self.capacity / self.max_iter) * self._iter
            capacity = tf.clip_by_value(capacity, 0, self.capacity)

            # calculate the reconstruction loss
            reconstruction_loss = self._reconstruction_loss_fn(x, reconstruction)

            # per sample and per latent dimension kl loss
            kl_loss = (
                1 + z_log_var - K.backend.square(z_mean) - K.backend.exp(z_log_var)
            )
            # take kl loss sum across latent dimensions
            kl_loss = -0.5 * K.backend.sum(kl_loss, axis=1)
            # then take the mean across latent dimensions
            kl_loss = K.backend.mean(kl_loss)

            # apply beta and capacity
            capacity_loss = self.gamma * K.backend.abs(kl_loss - capacity)
            total_loss = reconstruction_loss + capacity_loss

        # calculate and update gradients
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # increase training iteration
        self._iter.assign_add(1)

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "capacity_loss": capacity_loss,
            "capacity": capacity,
        }

    def call(self, x, **kwargs):
        if isinstance(x, tuple):
            x = x[0]
        z_mean, z_log_var, z = self.encode(x, **kwargs)
        reconstruction = self.decode(z, **kwargs)
        return reconstruction, z_mean

    def encode(self, x, **kwargs):
        """Encode an input using the probabilistic encoder.

        Parameters
        ----------
        x : tf.Tensor, np.ndarray (N, W, H, C)
            The input image data.

        Returns
        -------
        z_mean : (N, latent_dims)
            The mean latent value.
        z_log_var : (N, latent_dims)
            The log of the variance.
        z : (N, latent_dims)
            The sampled latents.
        """
        encoded = self.encoder(x, **kwargs)
        z_mean, z_log_var, z = self.sampler(encoded)
        return z_mean, z_log_var, z

    def decode(self, z, **kwargs):
        """Decode an input using the convolutional decoder.

        Parameters
        ----------
        z : tf.Tensor, np.ndarray (N, latent_dims)
            The (sampled) latent representation.

        Returns
        -------
        reconstruction : tf.Tensor, np.ndarray (N, W, H, C)
            The reconstructed image.
        """
        z_reshape = self.reshape(z)
        reconstruction = self.decoder(z_reshape, **kwargs)
        reconstruction = self._decoder_output(reconstruction)
        return reconstruction
