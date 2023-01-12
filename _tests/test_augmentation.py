from unittest.mock import patch

import numpy as np
import pytest
import tensorflow as tf

from cellx.augmentation.image import (
    augment_random_noise,
    augment_random_rot,
    augment_random_translation,
)


def test_augment_random_noise():
    const = tf.constant(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6, 0.7],
        ],
        dtype=tf.float32,
    )
    img = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    with patch(
        "tensorflow.random.normal",
        return_value=const[tf.newaxis, ..., tf.newaxis],
    ):
        augmented = augment_random_noise(img[tf.newaxis, ..., tf.newaxis])[
            0, ..., 0
        ]
    expected = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 0.1, 1.2, 0.3],
            [0.4, 0.5, 1.6, 1.7],
        ],
    )
    assert np.allclose(augmented, expected, atol=0.01)


expected_outputs_translation = [
    np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    ),
    np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    ),
    np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ],
        dtype=np.float32,
    ),
]


@pytest.mark.parametrize("n,dx,dy", [(0, 1, 3), (1, -1, 1), (2, 2, 0)])
def test_augment_random_translation(n, dx, dy):
    img = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 4.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    with patch("tensorflow.random.uniform", side_effect=[dx, dy]):
        augmented = augment_random_translation(
            img[tf.newaxis, ..., tf.newaxis]
        )[0, ..., 0]
    expected = expected_outputs_translation[n]
    assert np.allclose(augmented, expected, atol=0.01)


expected_outputs_rotation = [
    np.array(
        [
            [0.0, 0.0, 1.58, 1.31],
            [0.0, 0.52, 1.49, 0.52],
            [0.0, 0.56, 0.26, 0.0],
            [0.0, 0.14, 0.0, 0.0],
        ],
        dtype=np.float32,
    ),
    np.array(
        [
            [1.0, 3.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    ),
    np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0, 1.0],
        ],
        dtype=np.float32,
    ),
]


@pytest.mark.parametrize("n,angle", [(0, np.pi * (2 / 3)), (1, np.pi), (2, 0)])
def test_augment_random_rot(n, angle):
    img = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0, 1.0],
        ],
        dtype=np.float32,
    )
    with patch("tensorflow.random.uniform", return_value=angle):
        augmented = augment_random_rot(img[tf.newaxis, ..., tf.newaxis])[
            0, ..., 0
        ]
    expected = expected_outputs_rotation[n]
    assert np.allclose(augmented, expected, atol=0.01)
