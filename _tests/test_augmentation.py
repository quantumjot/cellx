import numpy as np
import tensorflow as tf

from cellx.augmentation.image import (
    augment_random_noise,
    augment_random_rot,
    augment_random_translation,
)


def test_augment_random_noise():
    tf.random.set_seed(0)
    img = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    augmented = augment_random_noise(img[tf.newaxis, ..., tf.newaxis])[
        0, ..., 0
    ]
    expected = np.array(
        [
            [0.24, 0.07, -0.07, -0.17],
            [-0.20, 0.08, 0.00, 0.19],
            [0.10, 0.10, -0.12, -0.07],
            [0.13, -0.12, 0.84, 0.85],
        ]
    )
    assert np.allclose(augmented, expected, atol=0.01)


def test_augment_random_translation():
    tf.random.set_seed(0)
    img = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    augmented = augment_random_translation(img[tf.newaxis, ..., tf.newaxis])[
        0, ..., 0
    ]
    expected = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    assert np.allclose(augmented, expected, atol=0.01)


def test_augment_random_rot():
    tf.random.set_seed(0)
    img = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    augmented = augment_random_rot(img[tf.newaxis, ..., tf.newaxis])[0, ..., 0]
    expected = np.array(
        [
            [0.0, 0.0, 0.49, 0.66],
            [0.0, 0.13, 0.85, 0.55],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert np.allclose(augmented, expected, atol=0.01)
