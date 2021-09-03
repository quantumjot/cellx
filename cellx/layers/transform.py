from typing import Optional

import tensorflow as tf
from tensorflow import keras as K


class PCATransform(K.layers.Layer):
    """Simple PCA Transform in Keras.

    Parameters
    ----------
    components : np.array, tf.Tensor (N, N)
        A square matrix of principal components to perform the transform. These
        are calculated using `sklearn.decomposition.PCA`.
    """

    def __init__(self, components, **kwargs):
        super().__init__(**kwargs)
        self.components = components

    def call(self, x, training: Optional[bool] = None):
        """Perform the transformation: T = XW"""
        T = tf.matmul(x, self.components)
        return T - tf.math.reduce_mean(T, axis=-2, keepdims=True)
