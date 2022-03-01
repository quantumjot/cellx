import numpy as np
import pytest
from sklearn.decomposition import PCA

from cellx.layers import ConvBlock2D, ConvBlock3D, PCATransform, ResidualBlock2D

LAYERS = [ConvBlock2D, ConvBlock3D, ResidualBlock2D]


@pytest.mark.parametrize("ndim", [1, 2, 4, 8])
def test_pca_transform(ndim: int):
    """Check that the PCA transform layer produces the same results as
    `scikit-learn`."""
    data = (np.random.randn(100, ndim) * 1.0 + 5.0).astype(np.float32)
    pca = PCA(n_components=ndim)
    x_true = pca.fit_transform(data)
    layer = PCATransform(pca.components_.T, pca.mean_)
    x_test = layer(data[np.newaxis, ...])[0, ...]
    np.testing.assert_almost_equal(x_true, x_test, decimal=5)


@pytest.mark.parametrize("ndim", [1, 2, 4, 8])
@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_pca_batch(ndim: int, batch_size: int):
    """Test a batch PCA transform."""
    raw = (np.random.randn(100, ndim) * 1.0 + 5.0).astype(np.float32)
    pca = PCA(n_components=ndim)
    pca.fit(raw)

    # set up the keras PCATransform layer
    layer = PCATransform(pca.components_.T, pca.mean_)

    # create a batch of data to transform (batch_size, 100, ndim)
    data = (np.random.randn(batch_size, 100, ndim) * 1.0 + 5.0).astype(np.float32)

    # iterate over each batch and transform using `scikit-learn`
    x_true = []
    for i in range(batch_size):
        batch = data[i, ...]
        x_true.append(pca.transform(batch))

    x_true = np.stack(x_true, axis=0)

    # perform the PCA transform as a batch operation
    x_test = layer(data)

    # make sure we have the correct shape and that the data match
    assert x_true.shape == x_test.shape
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
