import os
from typing import Callable, List, Optional, Tuple, Union

import dask
import dask.array as da
import numpy as np
from skimage.io import imread

# from skimage.transform import resize
# def _load_and_normalize(filename: str, output_shape: tuple = (64, 64)):
#     """Load an image, reshape to output_shape and normalize."""
#
#     # reshape to a certain image size
#     image = resize(imread(filename), output_shape, preserve_range=True)
#     n_pixels = np.prod(output_shape)
#     n_channels = image.shape[-1]
#
#     a_std = lambda d: np.max([np.std(d), 1.0 / np.sqrt(n_pixels)])
#     nrm = lambda d: np.clip((d - np.mean(d)) / a_std(d), -4.0, 4.0)
#
#     for dim in range(n_channels):
#         image[..., dim] = nrm(image[..., dim])
#
#     # TODO(arl): ????
#     image = np.clip(255.0 * ((image + 1.0) / 5.0), 0, 255)
#     return image


def _reader_based_on_extension(path: str) -> Callable:
    _, ext = os.path.splitext(path)
    if ext in [".npy", ".npz"]:
        return np.load
    else:
        return imread


def _coerce_shape(x, shape):
    pass


def _normalize(x):
    pass


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
        return da.from_array(data)

    if not isinstance(data, list):
        data = [data]

    # take a sample of the data
    sample = _reader_based_on_extension(data[0])(data[0])

    # now make a lazy dask array of the data
    images = [dask.delayed(_reader_based_on_extension(d))(d) for d in data]
    images = [
        da.from_delayed(x, shape=sample.shape, dtype=sample.dtype) for x in images
    ]
    images = da.stack(images, axis=0)

    return images
