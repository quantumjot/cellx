from typing import Optional, Union

import numpy as np


class InfinitePaddedImage:
    """InfinitePaddedImage

    Generates an infinitely padded image, by allowing indexing into negative
    regions or outside the bounding box of the original image.  Returns a user
    defined padding value (either a constant or mean of the image) when
    returning outside the original bounding box. Should handle any dimension of
    array, just for fun.

    Useful when dealing with Histogram of Oriented gradients for SVMs,
    Convolutional Neural Networks or anything that involves cropping a region
    of the image close to the border.

    Parameters
    ----------
    image : np.ndarray
        The input image data.
    mode : str
        'constant' or 'reflect'.
    pad_value : int, float optional
        Value to be used with the 'constant' mode.


    Returns
    -------
    padded_image : np.ndarray
        A pixel or slice of the original array

    """

    def __init__(
        self,
        image: np.ndarray,
        mode: str = "constant",
        pad_value: Optional[Union[int, float]] = None,
    ):
        if not isinstance(image, np.ndarray):
            raise TypeError

        self.pad_value = pad_value if pad_value is not None else np.mean(image)
        self._data = image
        self._mode = mode

    @property
    def shape(self):
        return self.data.shape

    @property
    def data(self) -> np.ndarray:
        return self._data

    def __getitem__(self, coords: tuple):
        if not isinstance(coords, tuple):
            raise TypeError

        if all([isinstance(c, slice) for c in coords]):
            return self._get_slice(coords)

        return self._get_pixel(coords)

    def _get_pixel(self, coords) -> np.ndarray:
        """Get a single pixel from the array."""
        if np.any(np.sign(coords) == -1) or np.any(
            [c > self.shape[i] - 1 for i, c in enumerate(coords)]
        ):
            return self.pad_value
        return self.data[coords]

    def _get_slice(self, r_coords) -> np.ndarray:
        """Get a multidimensional slice from the array."""

        # start by parsing the coordinates
        coords = self._parse(r_coords)

        # pad the image
        padded_image = self._pad(coords)

        if self._mode == "reflect":
            return padded_image

        # only need to do this for constant padded images
        trimmed_slice = []
        offsets = []
        for i, s in enumerate(coords):
            new_slice = [s.start, s.stop]
            if s.start < 0:
                new_slice[0] = 0
            if s.stop > self.shape[i]:
                new_slice[1] = self.shape[i]
            trimmed_slice.append(slice(new_slice[0], new_slice[1], None))
            offsets.append(
                slice(
                    new_slice[0] - s.start,
                    padded_image.shape[i] - (s.stop - new_slice[1]),
                    None,
                )
            )

        # get the cropped image and insert into the padded image
        cropped_image = self.data[tuple(trimmed_slice)]
        padded_image[tuple(offsets)] = cropped_image

        return padded_image

    def _parse(self, r_coords) -> list:
        """Parse the coordinates and return a full set."""
        # TODO: deal with Ellipsis

        coords = []
        for i, s in enumerate(r_coords):

            start = 0 if not s.start else s.start
            stop = self.shape[i] if not s.stop else s.stop
            coord = slice(start, stop, s.step)

            coords.append(coord)

        return coords

    def _pad(self, coords) -> np.ndarray:
        """Pad the image appropriately."""

        # if we're using a constant value, we're done!
        if self._mode == "constant":
            padded_im = np.ones(tuple([(s.stop - s.start) for s in coords]))
            return float(self.pad_value) * padded_im.astype(self.data.dtype)

        meshes = np.meshgrid(
            *[np.arange(s.start, s.stop) for s in coords], indexing="ij"
        )

        # use the X and y coords to determine the direction of the flips and
        # the position into the original data
        idx = []
        for dim, mesh in enumerate(meshes):
            dir = -2 * np.mod(mesh // self.shape[dim], 2) + 1.0  # convert to +1/-1
            offset = np.mod(mesh * dir, self.shape[dim])
            indices = offset.astype(np.int)
            idx.append(indices)

        return self.data[tuple(idx)]
