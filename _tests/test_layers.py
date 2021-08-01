import pytest

from cellx.layers import ConvBlock2D, ConvBlock3D, ResidualBlock2D


@pytest.mark.parametrize("layer", [ConvBlock2D, ConvBlock3D, ResidualBlock2D])
def test_layer_instantiation(layer):
    """Test instantiating network layers."""
    new_layer = layer()
    assert isinstance(new_layer, layer)
