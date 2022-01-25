import numpy as np
import pytest
from sklearn.decomposition import PCA

from cellx.layers import ConvBlock2D, ConvBlock3D, PCATransform, ResidualBlock2D

LAYERS = [ConvBlock2D, ConvBlock3D, ResidualBlock2D]


@pytest.mark.parametrize("ndim", [1, 2, 4, 8])
def test_pca_transform(ndim: int):
    """Check that the PCA transform layer produces the same results as
    scikit-learn."""
    data = (np.random.randn(100, ndim) * 1.0 + 5.0).astype(np.float32)
    pca = PCA(n_components=ndim)
    x_true = pca.fit_transform(data)
    layer = PCATransform(pca.components_.T, pca.mean_)
    x_test = layer(data[np.newaxis, ...])[0, ...]
    np.testing.assert_almost_equal(x_true, x_test, decimal=5)


@pytest.mark.parametrize("layer", LAYERS)
def test_layer_instantiation(layer):
    """Test instantiating network layers."""
    new_layer = layer()
    assert isinstance(new_layer, layer)


@pytest.mark.parametrize("layer", LAYERS)
def test_layer_serialization(layer):
    """Test `get_config()` method derived from `SerializationMixin`."""
    new_layer = layer()
    config = new_layer.get_config()
    assert isinstance(config, dict)


@pytest.mark.parametrize("layer", LAYERS)
@pytest.mark.parametrize("filters", [8, 16, 64])
def test_layer_serialization_dict(layer, filters):
    """Test `get_config()` method derived from `SerializationMixin` returns the
    value used to initialize the layer."""
    new_layer = layer(filters=filters)
    config = new_layer.get_config()
    assert "filters" in config.keys()
    assert config["filters"] == filters
