import io

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from scipy.special import expit as sigmoid
from scipy.special import softmax
from sklearn.metrics import confusion_matrix

from .tools.confusion import plot_confusion_matrix


def test_pred_binary(_pred):
    """Return the classification label from unscaled logits."""
    return np.round(sigmoid(_pred)).astype(np.int)


def test_pred_multiclass(_pred):
    """Return the classification label from unscaled logits."""
    return np.argmax(softmax(_pred, axis=-1), axis=-1).astype(np.int)


def tensorboard_montage_callback(model: K.Model, test_images: np.ndarray, logdir: str):
    """Create a callback that writes summary montage images to a tensorboard
    log. Useful while training networks that generate images as output.

    Parameters
    ----------
    model : Keras.Model
        The model.
    test_images : np.ndarray
        An array of images or volumes. First axis is batch.
    logdir : str
        Path to the tensorboard log directory.
    """

    file_writer_montage = tf.summary.create_file_writer(logdir + "/montage")

    def log_montage(epoch, logs):

        # make the model prediction
        x_hat = model.predict(test_images)

        # make montages and concatenate them
        x_montage = _plot_montage(test_images)
        x_hat_montage = _plot_montage(x_hat)
        summary_montage = tf.concat([x_montage, x_hat_montage], axis=1)

        with file_writer_montage.as_default():
            tf.summary.image("montage", summary_montage, step=epoch)

    # make a lambda call back
    return K.callbacks.LambdaCallback(on_epoch_end=log_montage)


def tensorboard_confusion_matrix_callback(
    model: K.Model,
    test_images: np.ndarray,
    test_labels: list,
    logdir: str,
    class_names: list = [],
    is_binary: bool = True,
):

    """Create a callback that writes a summary confusion matrix to a tensorboard
    log. Useful while training networks that perform classification.

    Parameters
    ----------
    model : Keras.Model
        The model.
    test_images : np.ndarray
        An array of images or volumes. First axis is batch.
    test_labels : list
        A list of labels, sparse categorical.
    logdir : str
        Path to the tensorboard log directory.
    class_names : str
        A list of the class names for each label.
    is_binary : bool
        Flag that determines whether the classification is binary or multiclass.

    Notes
    -----
    Modified from: https://www.tensorflow.org/tensorboard/image_summaries
    """

    file_writer_cm = tf.summary.create_file_writer(logdir + "/cm")

    # set the prediction function
    test_pred_fn = test_pred_binary if is_binary else test_pred_multiclass

    def log_confusion_matrix(epoch, logs):
        # Use the model to predict the values from the validation dataset.
        test_pred_raw = model.predict(test_images)
        test_pred = test_pred_fn(test_pred_raw)

        # Calculate the confusion matrix.
        cm = confusion_matrix(test_labels, test_pred)

        # Log the confusion matrix as an image summary.
        figure = plot_confusion_matrix(cm, class_names=class_names)
        cm_image = _plot_to_image(figure)

        # Log the confusion matrix as an image summary.
        with file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)

    # make a lambda call back
    return K.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)


def _plot_to_image(figure):
    """ converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def _plot_montage(x, max_images: int = 32, columns: int = 8, rows: int = 4):
    """ make a montage of the images """

    x = x[:max_images, ...]

    rgb = tf.stack([x[..., 1], x[..., 0], x[..., 1]], axis=-1)

    rgb = tf.pad(
        tensor=rgb,
        paddings=[[0, 0], [1, 1], [1, 1], [0, 0]],
        mode="CONSTANT",
        constant_values=tf.reduce_min(input_tensor=rgb),
    )

    # clip the outputs
    rgb = tf.clip_by_value(rgb, -3.0, 3.0)
    rgb = (rgb + 3.0) / 6.0

    montage = []
    for r in range(rows):
        j = columns * r
        row = tf.concat([rgb[i, ...] for i in range(j, j + columns)], axis=1)
        montage.append(row)

    montage = tf.concat(montage, axis=0)
    return tf.cast(montage[tf.newaxis, ...] * 255, tf.uint8)
