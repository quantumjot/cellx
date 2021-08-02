import itertools
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

import os

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    errors: Optional[np.ndarray] = None,
    figsize: Tuple[int] = (8, 8),
    normalize: bool = True,
):
    """Creates a matplotlib figure containing the confusion matrix.

    Parameters
    ----------
    cm : np.ndarray
        A confusion matrix of integer classes.
    class_names : list
        String names of the integer classes.
    errors : np.ndarray, optional
        The errors associated with each position in the confusion matrix.
    figsize : tuple, optional
        A tuple defining the figure size.
    normalize : bool, default = True
        A flag to normalize the entries in the confusion matrix.

    Returns
    -------
    figure : matplotlib.figure
        The figure.

    Notes
    -----
    Modified from: https://www.tensorflow.org/tensorboard/image_summaries
    """

    figure = plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("Confusion matrix must be a 2D square array")

    if len(class_names) != cm.shape[0]:
        raise ValueError(
            "Number of classes does not match dimensions of confusion matrix"
        )

    # normalize the confusion matrix
    if normalize:
        cm = np.around(cm.astype(np.float) / cm.sum(axis=1)[:, np.newaxis], decimals=2)
        plt.clim([0, 1])

    if errors is not None:
        if errors.shape != cm.shape:
            raise ValueError("Error shape does not match confusion matrix shape")

    txt_params = {
        "horizontalalignment": "center", "verticalalignment": "center",
    }

    # use white text if squares are dark; otherwise black
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        value = f"{cm[i, j]: .2f}"
        if errors is not None:
            value += f" \n \U000000B1 {errors[i, j]: .2f}"
        plt.text(j, i, value, color=color, **txt_params)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure
