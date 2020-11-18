import itertools
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    cm: np.ndarray, class_names: list, figsize: Tuple[int] = (8, 8)
):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Parameters
    ----------
    cm : np.ndarray
        A confusion matrix of integer classes.
    class_names : list
        String names of the integer classes.
    figsize : tuple, optional
        A tuple defining the figure size.

    Returns
    -------

    figure : matplotlib.figure
        The figure.

    Notes:
        modified from: https://www.tensorflow.org/tensorboard/image_summaries
    """

    figure = plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # normalize the confusion matrix
    cm = np.around(cm.astype(np.float) / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # use white text if squares are dark; otherwise black
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure
