import numpy as np
import pytest

from cellx.tools.projection import ManifoldProjection2D


def _make_square(shape: tuple = (64, 64)):
    img = np.zeros(shape, dtype=np.float32)
    r = np.random.randint(1, shape[0] // 2 - 1)
    s = slice(shape[0] // 2 - r, shape[0] // 2 + r, 1)
    img[s, s] = 1
    return img


def _make_circle(shape: tuple = (64, 64)):
    img = np.zeros(shape, dtype=np.float32)
    r = np.random.randint(1, shape[0] // 2 - 1)
    xr = np.linspace(-shape[0] / 2, shape[0] / 2, shape[0])
    xx, yy = np.meshgrid(xr, xr)
    d = np.sqrt(xx * xx + yy * yy)
    img[d < r] = 1
    return img


def _make_data(n_images: int = 1_000, shape: tuple = (64, 64)):
    """Make some fake data for the manifold projection."""

    n_squares = n_images // 2
    n_circles = n_squares

    cxy = np.random.normal(-3.0, 1.0, size=(n_circles, 2))
    sxy = np.random.normal(3.0, 1.0, size=(n_squares, 2))

    imgs = [_make_square(shape=shape) for _ in range(n_squares)] + [
        _make_circle(shape=shape) for _ in range(n_circles)
    ]
    imgs = np.stack(imgs, axis=0)[..., np.newaxis]
    xy = np.concatenate([sxy, cxy], axis=0)

    return imgs, xy


@pytest.mark.parametrize("method", ["mean", "max", "sum", "first"])
def test_manifold_method(method, shape=(64, 64)):
    imgs, xy = _make_data(shape=shape)
    manifold = ManifoldProjection2D(imgs, output_shape=shape)
    proj, _, _ = manifold(xy, method=method)
