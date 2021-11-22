import matplotlib.pyplot as plt
import numpy as np

from cellx.tools.confusion import plot_confusion_matrix


def test_confusion_matrix():
    data = np.random.random((5, 5))
    labels = [f"label-{i}" for i in range(data.shape[0])]
    fig = plot_confusion_matrix(data, labels)
    assert isinstance(fig, plt.Figure)
