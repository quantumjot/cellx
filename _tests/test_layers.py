import numpy as np
import pytest
from sklearn.decomposition import PCA

from cellx.layers import ConvBlock2D, ConvBlock3D, PCATransform, ResidualBlock2D


@pytest.mark.parametrize("layer", [ConvBlock2D, ConvBlock3D, ResidualBlock2D])
def test_layer_instantiation(layer):
    """Test instantiating network layers."""
    new_layer = layer()
    assert isinstance(new_layer, layer)


@pytest.mark.parametrize("ndim", [1, 2, 4, 8])
def test_pca_transform(ndim: int):
    """Check that the PCA transform layer produces the same results as
    scikit-learn."""
    data = np.random.randn(100, ndim) * 1.0 + 5.0
    pca = PCA(n_components=ndim)
    pca.fit(data)
    layer = PCATransform(pca.components_, pca.mean_)
    x_true = pca.transform(data.astype(np.float32))
    x_test = np.squeeze(layer(data[np.newaxis, ...]))
    np.testing.assert_almost_equal(x_true, x_test, decimal=5)
