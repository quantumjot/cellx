from functools import partial
from typing import Tuple, Union

import numpy as np
from scipy.stats import binned_statistic_2d
from tqdm import tqdm

from ..io import lazy_load_images
from ..utils import CallableEnum


class ProjectionMethod(CallableEnum):
    MEAN = partial(np.mean, axis=0)
    MAX = partial(np.max, axis=0)
    MIN = partial(np.min, axis=0)
    SUM = partial(np.sum, axis=0)
    STD = partial(np.std, axis=0)
    FIRST = partial(lambda x: x[0, ...])


class ManifoldProjection2D:
    """ManifoldProjection2D.

    Make a montage of image patches that represent examples from a manifold
    projection.  Uses lazy loader to deal with image data.

    Parameters
    ----------
    images : list of str or (N, W, H, C) np.ndarray
        A list of image filenames or a numpy array of N images, width W, height
        H, and C channels.
    output_shape : tuple of int
        Final size to reshape individual image patches to for the montage.
    """

    def __init__(self, images: list, output_shape: tuple = (64, 64)):
        self._output_shape = output_shape
        self._images = lazy_load_images(images)

        if self._images.ndim != 4:
            raise ValueError(
                f"Image data should be at least NWHC. Found shape: {self._images.shape}."
            )

    def __call__(
        self,
        manifold: np.ndarray,
        bins: int = 32,
        components: Tuple[int] = (0, 1),
        method: Union[str, ProjectionMethod] = "mean",
    ) -> tuple:
        """Build the projection.

        Parameters
        ----------
        manifold : np.ndarray
            Numpy array of the manifold projection.
        bins : int
            Number of two-dimensional bins to group the manifold examples in.
        components : tuple of int
            Dimensions of manifold to use when creating the projection.
        method : str or ProjectionMethod, default = 'mean'
            A method to collapse samples when generating the final image.
                * MEAN - the per-pixel mean of the images in the bin
                * STD - the per-pixel standard deviation of images in the bin
                * FIRST - the first image in the bin

        Returns
        -------
        imgrid : np.ndarray
            An image with example image patches from the manifold arranged on a
            grid.
        counts : np.ndarray
            A 2d histogram of the number of image patches per bin. Note that
            this is on a different scale to the imgrid.
        extent : tuple
            Delimits the minimum and maximum bin edges, in each dimension, used
            to create the result.
        """

        assert manifold.shape[0] == len(self._images)
        assert manifold.shape[-1] >= len(components)
        assert self._images.ndim == 4

        # get the correct function for projecting the images
        if isinstance(method, str):
            method = ProjectionMethod[method.upper()]

        # bin the manifold
        counts, xe, ye, bn = binned_statistic_2d(
            manifold[:, components[0]],
            manifold[:, components[1]],
            [],
            bins=bins,
            statistic="count",
            expand_binnumbers=True,
        )

        bxy = zip(bn[0, :].tolist(), bn[1, :].tolist())

        # make a lookup dictionary
        grid = {}
        for idx, b in enumerate(bxy):
            if b not in grid:
                grid[b] = []
            grid[b].append(self._images[idx, ...])

        # now make the grid image
        full_bins = [int(b) for b in self._output_shape]
        half_bins = [b // 2 for b in self._output_shape]
        imgrid = np.zeros(
            (
                (full_bins[0] + 1) * bins + half_bins[0],
                (full_bins[1] + 1) * bins + half_bins[1],
                self._images.shape[-1],
            ),
            dtype=self._images.dtype,
        )

        # build it
        for xy, images in tqdm(grid.items()):

            stack = np.stack(images, axis=0)
            im = method(stack)

            xx, yy = xy
            blockx = slice(
                xx * full_bins[0] - half_bins[0],
                xx * full_bins[0] - half_bins[0] + self._output_shape[0],
                1,
            )
            blocky = slice(
                yy * full_bins[1] - half_bins[1],
                yy * full_bins[1] - half_bins[1] + self._output_shape[1],
                1,
            )

            imgrid[blockx, blocky, :] = im

        extent = (min(xe), max(xe), min(ye), max(ye))

        return imgrid, counts, extent
