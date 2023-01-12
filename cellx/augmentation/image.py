import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

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


@augmentation_label_handler
def augment_random_noise(x: tf.Tensor) -> tf.Tensor:
    """Perform a noise addition augmentation.

    Parameters
    ----------
    x : tf.Tensor (T, ...)
        The tensor to be augmented.

    Returns
    -------
    x : tf.Tensor
        The augmented tensor.
    """
    stddev = tf.math.reduce_std(x) / 2
    x_add = tf.random.normal(shape=tf.shape(x), stddev=stddev)
    x = tf.add(x, x_add)
    return x


@augmentation_label_handler
def augment_random_translation(x: tf.Tensor) -> tf.Tensor:
    """Perform a random translation augmentation.

    The image contents are shifted by a random x/y shift. The empty space left
    behind is filled with zeros, while the image contents that are shifted out
    of the field of view are discarded.

    Parameters
    ----------
    x : tf.Tensor (T, ...)
        The tensor to be augmented.

    Returns
    -------
    x : tf.Tensor
        The augmented tensor.
    """
    w, h = tf.shape(x)[-3], tf.shape(x)[-2]
    dx = tf.random.uniform(
        shape=(), minval=-w // 2, maxval=w // 2, dtype=tf.int32
    )
    dy = tf.random.uniform(
        shape=(), minval=-h // 2, maxval=h // 2, dtype=tf.int32
    )
    x = tfa.image.translate(x, [dx, dy])
    return x


@augmentation_label_handler
def augment_random_rot(x: tf.Tensor) -> tf.Tensor:
    """Perform a random rotation augmentation where the image(s) are
    rotated by a fixed angle, uniformly sampled from the range [0.,2*pi).

    Parameters
    ----------
    x : tf.Tensor (T, ...)
        The tensor to be augmented.

    Returns
    -------
    x : tf.Tensor
        The augmented tensor.
    """
    angle = tf.random.uniform(shape=(), maxval=2 * np.pi)
    x = tfa.image.rotate(x, angle, "bilinear", "reflect")
    return x
