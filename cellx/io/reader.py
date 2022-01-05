import os
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import dask
import dask.array as da
import numpy as np
from skimage.io import imread
from skimage.transform import resize


def _reader_based_on_extension(path: str) -> Callable:
    """Returns a reader function based on the filetype."""
    _, ext = os.path.splitext(path)
    if ext in [".npy", ".npz"]:
        return np.load
    else:
        return imread


def _coerce_shape(x, shape):
    """Coerce the shape of the image data by resizing the image."""
    return resize(x, shape, preserve_range=True)


def _normalize(x, shape):
    """Perform normalization on the image."""
    n_pixels = np.prod(shape)
    n_channels = x.shape[-1]

    a_std = lambda d: np.max([np.std(d), 1.0 / np.sqrt(n_pixels)])
    nrm = lambda d: np.clip((d - np.mean(d)) / a_std(d), -4.0, 4.0)

    for dim in range(n_channels):
        x[..., dim] = nrm(x[..., dim])

    # TODO(arl): this scales the data back to uint8(ish)
    x = np.clip(255.0 * ((x + 1.0) / 5.0), 0, 255)
    return x


def _reader(
    path: os.PathLike, shape: Optional[Tuple[int]] = None, normalize: bool = True
) -> Callable:
    """Create a reader function which will normalize and coerce shape."""

    def reader_fn(
        path: os.PathLike, shape: Optional[Tuple[int]] = None, normalize: bool = False
    ):

        data = _reader_based_on_extension(path)(path)

        if shape is not None:
            data = _coerce_shape(data, shape)
        if normalize:
            data = _normalize(data, shape)
        return data

    return partial(reader_fn, shape=shape, normalize=normalize)


def lazy_load_images(
    data: Union[os.PathLike, np.ndarray, List[os.PathLike]],
    normalize: bool = True,
    coerce_shape: Optional[Tuple[int]] = None,
) -> dask.array:
    """Load image data into a dask array for processing.

    Parameters
    ----------
    data : PathLike, array, List[PathLike]
        The data to be loaded.
    normalize : bool
        Flag to determine whether to normalize the data.
    coerce_shape : tuple, None
        Whether to coerece the images to a certain shape.

    Returns
    -------
    images : dask.array
        A dask array of the format N, (Z), W, H, C. Lazy loading.
    """

    if isinstance(data, np.ndarray):
        return da.from_array(data, chunks="auto")

    if not isinstance(data, list):
        data = [data]

    # take a sample of the data
    reader = _reader(data, shape=coerce_shape, normalize=normalize)
    sample = reader(data[0])

    # now make a lazy dask array of the data
    images = [dask.delayed(reader)(d) for d in data]
    images = [
        da.from_delayed(x, shape=sample.shape, dtype=sample.dtype) for x in images
    ]
    images = da.stack(images, axis=0)

    return images
