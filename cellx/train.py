from typing import Union

import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def compute_class_weight_dict(
    labels: Union[list, np.ndarray], class_weight: Union[dict, str, None] = "balanced"
) -> dict:
    """Compute class weight.

    Wrapper for sklean function that returns Keras compatible dictionary.

    Parameters
    ----------
    labels : list, np.ndarray
        The array of labels to be balanced.
    class_weight : dict, str, None
        Additional parameter for sklearn.compute_class_weight

    Returns
    -------
    class_weights : dict
        A keras compatible dictionary of class weights. Keys are the labels,
        and the values are the weightings.
    """
    unique_labels = np.unique(labels)
    _weight = compute_class_weight(class_weight, classes=unique_labels, y=labels)
    return {unique_labels[k]: w for k, w in enumerate(_weight)}
