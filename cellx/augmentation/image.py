import numpy as np
import tensorflow as tf

from .utils import augmentation_label_handler


@augmentation_label_handler
def augment_random_boundary(x: tf.Tensor) -> tf.Tensor:
    """Perform a random cropping type augmentation to simulate the edge of a
    field of view.

    Parameters
    ----------
    x : tf.Tensor (T, ...)
        The tensor to be augmented.

    Returns
    -------
    x : tf.Tensor
        The augmented tensor.
    """

    vignette = np.ones(x.shape, dtype=np.float32)
    width = np.random.randint(0, x.shape[1] // 2)
    vignette[:, :width, ...] = 0
    x = tf.multiply(x, vignette)
    return x


@augmentation_label_handler
def augment_random_rot90(x: tf.Tensor) -> tf.Tensor:
    """Perform a random rotation type augmentation.

    Parameters
    ----------
    x : tf.Tensor (T, ...)
        The tensor to be augmented.

    Returns
    -------
    x : tf.Tensor
        The augmented tensor.
    """
    k = tf.random.uniform(maxval=3, shape=(), dtype=tf.int32)
    x = tf.image.rot90(x, k=k)
    return x


@augmentation_label_handler
def augment_random_flip(x: tf.Tensor) -> tf.Tensor:
    """Perform a random flipping type augmentation.

    Parameters
    ----------
    x : tf.Tensor (T, ...)
        The tensor to be augmented.

    Returns
    -------
    x : tf.Tensor
        The augmented tensor.
    """
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    return x
