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


def load_model(filename: str, extra_custom_objects: dict = {}, **kwargs) -> K.Model:
    """Convenience function to load a Keras model with custom cellx layers.

    Parameters
    ----------
    filename : str
        Filename for the model.
    extra_custom_objects : dict
        A dictionary containing any extra custom objects for the model.

    Returns
    -------
    model : keras.Model
        The model.
    """
    custom_objects = _get_custom_layers_dict()

    if not isinstance(extra_custom_objects, dict):
        raise TypeError(
            f"`extra_custom_objects` should be a dictionary not "
            f"{type(extra_custom_objects)}."
        )
    custom_objects.update(extra_custom_objects)

    model = K.models.load_model(filename, custom_objects=custom_objects, **kwargs)
    return model
