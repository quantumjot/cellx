import tensorflow as tf
import tensorflow.keras as K
import numpy as np

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
        x_montage = montage(test_images)
        x_hat_montage = montage(x_hat)
        summary_montage = tf.concat([x_montage, x_hat_montage], axis=1)

        with file_writer_montage.as_default():
            tf.summary.image('montage', summary_montage, step=epoch)

    # make a lambda call back
    callback = K.callbacks.LambdaCallback(on_epoch_end=log_montage)

    return callback



def montage(x,
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
