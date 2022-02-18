import os
from pathlib import Path

import numpy as np

from cellx.tools.dataset import write_dataset

# import pytest


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
