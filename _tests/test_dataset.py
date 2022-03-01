import os
from pathlib import Path

import numpy as np
import pytest

from cellx.tools.dataset import (
    build_dataset,
    count_images_in_dataset,
    list_tfrecord_files,
    write_dataset,
)


def _make_images(n_images: int = 10, shape: tuple = (64, 64, 1)):
    """Generate a stack of random image data."""
    rng = np.random.default_rng()
    return rng.integers(0, 255, size=(n_images,) + shape).astype(np.uint8)


def _test_write_images(filename, dtype="uint8"):
    """Make some images and write them out."""
    images = _make_images().astype(dtype)
    write_dataset(filename, images)
    return images


def test_write_dataset(tmp_path):
    """Test writing out a dataset."""
    filename = Path(tmp_path) / "test.tfrecord"
    _ = _test_write_images(filename)

    # make sure the exported filename exists
    assert os.path.exists(filename)


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_read_write_dataset_dtype(tmp_path, dtype):
    """Test to make sure serialized data matches raw data."""
    # note(arl) - this function doesn't work with a path as input!!
    filename = Path(tmp_path) / "test.tfrecord"
    images = _test_write_images(filename, dtype=dtype)

    # build the dataset and return the first record
    ds = build_dataset(tmp_path, output_dtype=dtype)
    x = list(ds.take(1).as_numpy_iterator())[0]

    # make sure we have the correct shape
    assert x.shape == images.shape[1:]

    # make sure the data matches
    np.testing.assert_equal(x, images[0, ...])


def test_list_tfrecords(tmp_path):
    """Check that list_tfrecords returns a list of tfrecord files when using
    different inputs, e.g. single file, folder or list."""
    filename = Path(tmp_path) / "test.tfrecord"
    _ = _test_write_images(filename)

    # single path to file
    files = list_tfrecord_files(filename)
    assert files == [filename]

    # list of files
    files = list_tfrecord_files([filename])
    assert files == [filename]

    # path as string
    files = list_tfrecord_files(tmp_path)
    assert files == [filename]

    # path as `Path`
    files = list_tfrecord_files(Path(tmp_path))
    assert files == [filename]


@pytest.mark.parametrize("n_images", [1, 10, 100])
def test_count_images(tmp_path, n_images):
    """Make datasets with varying numbers of images. Test counting the number
    of images in the dataset."""
    filename = Path(tmp_path) / "test.tfrecord"
    images = _make_images(n_images=n_images)
    write_dataset(filename, images)

    # count the number of images in the dataset
    n = count_images_in_dataset(filename)
    assert n == n_images
