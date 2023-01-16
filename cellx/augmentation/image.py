import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from .utils import augmentation_label_handler


def _form_vignette(x: tf.Tensor, k: int, crop_fraction: float):
    axis = k % 2
    width = tf.cast(
        crop_fraction * tf.cast(tf.shape(x)[axis], tf.float32), tf.int32
    )
    residue_shape = tf.gather(tf.shape(x), [1 + (axis % -2), 2])
    vignette = tf.concat(
        [
            tf.zeros(tf.concat([[width], residue_shape], axis=0)),
            tf.ones(
                tf.concat([[tf.shape(x)[axis] - width], residue_shape], axis=0)
            ),
        ],
        axis=0,
    )
    vignette = tf.image.rot90(vignette, k=k)
    return vignette


@augmentation_label_handler
def augment_random_boundary(
    x: tf.Tensor, max_crop_fraction: float = 0.5
) -> tf.Tensor:
    """Perform a random cropping type augmentation to simulate the edge of a
    field of view.

    Applies cropping operation independently to each image in the stack.

    Parameters
    ----------
    x : tf.Tensor (T, ...)
        The tensor to be augmented.

    Returns
    -------
    x : tf.Tensor
        The augmented tensor.
    """
    crop_fractions = tf.random.uniform(
        shape=[tf.shape(x)[0]], maxval=max_crop_fraction, dtype=tf.float32
    )
    rotations = tf.random.uniform(
        shape=[tf.shape(x)[0]], maxval=4, dtype=tf.int32
    )
    vignettes = tf.map_fn(
        lambda i: _form_vignette(i[0], i[1], i[2]),
        (x, rotations, crop_fractions),
        fn_output_signature=tf.float32,
    )
    x = tf.multiply(x, vignettes)
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
def augment_random_noise(x: tf.Tensor, stddev: float = 0.5) -> tf.Tensor:
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
