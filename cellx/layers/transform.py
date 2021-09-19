from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras as K


class PCATransform(K.layers.Layer):
    """Simple PCA Transform in Keras.

    Parameters
    ----------
    components : np.array, tf.Tensor (N, N)
        A square matrix of principal components to perform the transform. These
        are calculated using `sklearn.decomposition.PCA`.

    Notes
    -----
    Assumes that the number of principal components is the same as the number of
    input features, i.e. that the components matrix is square.
    """

    def __init__(self, components: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.components = components
        self.n_components = components.shape[-1]

    def call(self, x, training: Optional[bool] = None):
        """Perform the transformation: T = XW"""
        T = tf.reshape(
            tf.matmul(tf.reshape(x, (-1, self.n_components)), self.components),
            (-1, x.shape[1], self.n_components),
        )
        return T - tf.math.reduce_mean(T, axis=1, keepdims=True)
