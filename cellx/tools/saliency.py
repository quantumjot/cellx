import numpy as np
import tensorflow as tf
from tensorflow import keras as K


def visualize_saliency(
    model: K.Model,
    data_wrt: np.ndarray,
    n_samples: int = 1,
    epsilon: float = 1e-18,
    normalize: bool = True,
    stddev_spread: float = 0.15,
):
    """Visualize the saliency wrt to an input tensor.

    Can visualize either the backprop gradients or use SmoothGrad to inject
    noise and visualize a sample of gradients.

    Parameters
    ----------
    model : Keras.Model
        The trained Keras model.
    data_wrt : np.ndarray, (1, ...)
        The data to calculate the gradients wrt. Batch size (i.e) first dimension
        should be one.
    n_samples : int
        The number of samples to take when generating the saliency. Must be greater than one.
    epsilon : float
        A small offset to prevent divide by zero when normalizing the gradients.
    normalize : bool
        Flag to specify whether to normalize the gradients in the range [0, 1].
    stddev_spread : float
        The standard deviation of the Gaussian noise used when sampling the gradients.

    Returns
    -------
    normalized_gradient : np.array
        The normalized gradients/saliency map.
    logits : tf.Tensor
        The output logits of the model.

    Notes
    -----
    Deep Inside Convolutional Networks: Visualising Image Classification
    Models and Saliency Maps
    Karen Simonyan and Andrea Vedaldi and Andrew Zisserman
    """

    # batch size should only be one
    assert data_wrt.shape[0] == 1
    assert n_samples > 0

    # SmoothGrad
    stddev = stddev_spread * (np.max(data_wrt) - np.min(data_wrt))

    # use the batch to perform the sampling
    tensor_wrt = np.concatenate([data_wrt] * n_samples, axis=0)
    assert tensor_wrt.ndim == data_wrt.ndim
    assert tensor_wrt.shape[0] == n_samples

    # SmoothGrad
    if n_samples > 1:
        rng = np.random.default_rng()
        noise = rng.normal(loc=0, scale=stddev, size=tensor_wrt.shape)
        tensor_wrt = tensor_wrt + noise

    # calculate the gradients using the sample
    tensor_wrt = tf.Variable(tensor_wrt, dtype=tf.float32)
    with tf.GradientTape() as tape:
        logits = model(tensor_wrt, training=False)
        # get the gradient and magnitude of the gradient
        gradient = tape.gradient(logits, tensor_wrt)
        square_gradient = tf.square(gradient)

    absolute_gradient = tf.reduce_mean(
        square_gradient,
        axis=0,
        keepdims=True,
    ).numpy()

    # if we're not normalizing, return the raw gradient magnitude
    if not normalize:
        return absolute_gradient, logits

    def _normalize_gradients(g):
        return (g - np.min(g)) / (np.max(g) - np.min(g) + epsilon)

    # now normalize the gradients
    normalized_gradient = _normalize_gradients(absolute_gradient)
    return normalized_gradient, logits
