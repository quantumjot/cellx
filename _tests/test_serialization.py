import numpy as np
import pytest
import tensorflow.keras as K

from cellx import load_model

from .test_layers import LAYERS


@pytest.mark.parametrize("layer", LAYERS)
def test_model_serialization(tmp_path, layer):
    """Test the model serialization. Creates a simple model using a specified
    layer then attempts to save and load the model. Compares the contents of the
    models to check proper serialization."""

    # model filename
    model_fn = tmp_path / "test_model.h5"

    # build a very simple network
    x = K.layers.Input(shape=(8, 8, 1))
    net = layer(filters=4)(x)
    y = K.layers.Softmax(axis=-1)(net)
    model = K.Model(inputs=x, outputs=y)
    model.compile(
        optimizer="rmsprop",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # save it out
    model.save(model_fn)

    # load it again
    new_model = load_model(model_fn)

    # now compare the layer types and weights
    for layer_i, layer_j in zip(model.layers, new_model.layers):

        # test the model layer types are the same
        assert type(layer_i) == type(layer_j)

        # test that the serialized model weights are the same
        for weights_i, weights_j in zip(
            layer_i.weights.numpy(), layer_j.weights.numpy()
        ):
            np.testing.assert_equal(weights_i.shape, weights_j.shape)
            np.testing.assert_equal(weights_i, weights_j)
