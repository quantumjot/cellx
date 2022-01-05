import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from tqdm import tqdm


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
    data_wrt : np.ndarray
        The data to calculate the gradients wrt.
    n_samples : int
        The number of samples to take when generating the saliency.
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

    # SmoothGrad
    stddev = stddev_spread * (np.max(data_wrt) - np.min(data_wrt))

    tensor_wrt = tf.Variable(data_wrt, dtype=tf.float32)

    # SmoothGrad
    if n_samples > 1:

        smooth_gradient = []
        for sample in tqdm(range(n_samples)):
            noise = tf.random.normal(data_wrt.shape, mean=0, stddev=stddev)
            tensor_plus_noise = tensor_wrt + noise

            with tf.GradientTape() as tape:
                logits = model(tensor_plus_noise, training=False)
                tape.watch(tensor_plus_noise)
                gradient = tape.gradient(logits, tensor_plus_noise)
                # absolute_gradient = np.abs(gradient.numpy())
                absolute_gradient = tf.square(gradient).numpy()

            smooth_gradient.append(absolute_gradient)
        absolute_gradient = (
            np.sum(np.stack(smooth_gradient, axis=0), axis=0) / n_samples
        )

    # gradients only
    else:
        with tf.GradientTape() as tape:
            logits = model(tensor_wrt, training=False)
            # get the gradient and magnitude of the gradient
            gradient = tape.gradient(logits, tensor_wrt)

        absolute_gradient = np.abs(gradient.numpy())

    # if we're not normalizing, return the raw gradient magnitude
    if not normalize:
        return absolute_gradient, logits

    def _normalize_gradients(g):
        return (g - np.min(g)) / (np.max(g) - np.min(g) + epsilon)

    # now normalize the gradients
    normalized_gradient = _normalize_gradients(absolute_gradient)
    return normalized_gradient, logits
