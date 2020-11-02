import inspect

from tensorflow import keras as K


def _get_custom_layers_dict() -> dict:
    """Find custom cellx layers and return a dictionary with the layer name and
    layer.

    Returns
    -------
    custom_objects : dict[str, layer]

    """
    from . import layers

    objects = inspect.getmembers(layers, inspect.isclass)
    custom_objects = {k: v for k, v in objects if issubclass(v, K.layers.Layer)}
    return custom_objects


def load_model(filename: str, **kwargs) -> K.Model:
    """Convenience function load a Keras model with custom cellx layers.

    Parameters
    ----------
    filename : str
        Filename for the model.

    Returns
    -------
    model : keras.Model
        The model.
    """
    custom_objects = _get_custom_layers_dict()
    model = K.models.load_model(filename, custom_objects=custom_objects, **kwargs)
    return model
