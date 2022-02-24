import os
from pathlib import Path

import numpy as np
import pytest

from cellx.tools.dataset import build_dataset, write_dataset


def _make_images(n_images: int = 10, shape: tuple = (64, 64, 1)):
    """Generate a stack of random image data."""
    rng = np.random.default_rng()
    return rng.integers(0, 255, size=(n_images,) + shape).astype(np.uint8)


def test_write_dataset(tmp_path):
    """Test writing out a dataset."""
    images = _make_images()

    # note(arl) - this function doesn't work with a path as input!!
    filename = Path(tmp_path) / "test.tfrecord"
    write_dataset(str(filename), images)

    # make sure the exported filename exists
    assert os.path.exists(filename)


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_read_write_dtype(tmp_path, dtype):
    """Test to make sure serialized data matches raw data."""
    images = _make_images().astype(dtype)

    # note(arl) - this function doesn't work with a path as input!!
    filename = Path(tmp_path) / "test.tfrecord"
    write_dataset(str(filename), images)

    # build the dataset and return the first record
    ds = build_dataset(tmp_path, output_dtype=dtype)
    x = list(ds.take(1).as_numpy_iterator())[0]

    # make sure we have the correct shape
    assert x.shape == images.shape[1:]

    # make sure the data matches
    np.testing.assert_equal(x, images[0, ...])
