from functools import wraps

import tensorflow as tf


def augmentation_label_handler(augmentation_fn):
    """Deals with functions that also return a label, by augmenting the data,
    but not the label (if present)."""

    @wraps(augmentation_fn)
    def _wrapper(*args, **kwargs):
        if len(args) == 2:
            data, label = args
            return augmentation_fn(data, **kwargs), label
        else:
            return augmentation_fn(*args, **kwargs)

    return _wrapper


def append_conditional_augmentation(
    dataset: tf.data.Dataset, augmentations: list, accept_probability: float = 0.1
):
    """Append augmentations to a TF dataset, each with a probability of
    acceptance."""
    for augmentation in augmentations:
        dataset.map(
            lambda x: tf.cond(
                tf.random.uniform([], 0.0, 1.0) > accept_probability,
                lambda: augmentation(x),
                lambda: x,
            )
        )
    return dataset
