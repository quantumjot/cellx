from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras as K

from .base import SerializationMixin


class PCATransform(K.layers.Layer, SerializationMixin):
    """Simple PCA Transform in Keras.

    Parameters
    ----------
    components : array, tf.Tensor (N, N)
        A square matrix of principal components to perform the transform. These
        are calculated using `sklearn.decomposition.PCA`.
    mean : array, tf.Tensor (N, )
        An array representing the per-feature empirical mean, estimated from the
        training set.

    Notes
    -----
    Assumes that the number of principal components is the same as the number of
    input features, i.e. that the components matrix is square.
    """

    def __init__(self, components: np.ndarray, mean: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.components = components
        self.mean = mean
        self.n_components = components.shape[-1]

        self._config = {"components": components, "mean": mean}

    def call(self, x, training: Optional[bool] = None):
        """Perform the transformation: T = (X-mu)W"""
        T = tf.reshape(
            tf.matmul(
                tf.reshape(x, (-1, self.n_components)) - self.mean, self.components
            ),
            (-1, x.shape[1], self.n_components),
        )
        return T
