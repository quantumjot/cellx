import io
import itertools

import tensorflow as tf
import tensorflow.keras as K
import numpy as np
from sklearn.metrics import confusion_matrix


import matplotlib.pyplot as plt

def tensorboard_montage_callback(model: K.Model,
                                 test_images: np.ndarray,
                                 logdir: str):

    """ create a callback that writes summary montage images to a tensorboard
    log. Useful while training networks that generate images as output """

    file_writer_montage = tf.summary.create_file_writer(logdir + '/montage')

    def log_montage(epoch, logs):

        # make the model prediction
        x_hat = model.predict(test_images)

        # make montages and concatenate them
        x_montage = plot_montage(test_images)
        x_hat_montage = plot_montage(x_hat)
        summary_montage = tf.concat([x_montage, x_hat_montage], axis=1)

        with file_writer_montage.as_default():
            tf.summary.image('montage', summary_montage, step=epoch)

    # make a lambda call back
    return K.callbacks.LambdaCallback(on_epoch_end=log_montage)





def tensorboard_confusion_matrix_callback(model: K.Model,
                                          test_images: np.ndarray,
                                          test_labels: list,
                                          logdir: str,
                                          class_names: list = []):

    """ create a callback that writes summary confusin matrix to a tensorboard
    log. Useful while training networks that performs classification

    Notes:
        modified from: https://www.tensorflow.org/tensorboard/image_summaries

        TODO(arl): THIS IS FOR BINARY CLASSIFCATION ONLY
    """

    file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')

    def _sigmoid(x):
        return 1. / (1. + np.exp(-x))

    def log_confusion_matrix(epoch, logs):
        # Use the model to predict the values from the validation dataset.
        test_pred_raw = model.predict(test_images)
        test_pred = np.round(_sigmoid(test_pred_raw)).astype(np.int)

        # Calculate the confusion matrix.
        cm = confusion_matrix(test_labels, test_pred)

        # Log the confusion matrix as an image summary.
        figure = plot_confusion_matrix(cm, class_names=class_names)
        cm_image = plot_to_image(figure)

        # Log the confusion matrix as an image summary.
        with file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)

    # make a lambda call back
    return K.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)




def plot_to_image(figure):
    """ converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image




def plot_montage(x,
                 max_images: int = 32,
                 columns: int = 8,
                 rows: int = 4):
    """ make a montage of the images """

    x = x[:max_images,...]

    rgb = tf.stack([x[...,1], x[...,0], x[...,1]], axis=-1)
    rgb = tf.pad(tensor=rgb,
                 paddings=[[0,0], [1,1], [1,1], [0,0]],
                 mode='CONSTANT',
                 constant_values=tf.reduce_min(input_tensor=rgb))

    # clip the outputs
    rgb = tf.clip_by_value(rgb, -3., 3.)
    rgb = (rgb + 3.) / 6.

    montage = []
    for r in range(rows):
        j = columns * r
        row = tf.concat([rgb[i,...] for i in range(j,j+columns)], axis=1)
        montage.append(row)

    montage = tf.concat(montage, axis=0)
    return tf.cast(montage[tf.newaxis,...]*255, tf.uint8)




def plot_confusion_matrix(cm: np.ndarray,
                          class_names: list):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes

    Notes:
        modified from: https://www.tensorflow.org/tensorboard/image_summaries
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure
