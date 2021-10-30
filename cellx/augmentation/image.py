import tensorflow as tf

from .utils import augmentation_label_handler


@augmentation_label_handler
def augment_boundary(x: tf.Tensor) -> tf.Tensor:
    """Perform a random cropping type augmentation to simulate the edge of a
    field of view.

    Parameters
    ----------

    Returns
    -------
    """
    return x
