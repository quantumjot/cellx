import numpy as np
import tensorflow as tf

from .handler import augmentation_label_handler



@augmentation_label_handler
def augment_timeseries_dummy(x: tf.Tensor):
    """ dummy augmention, returns original tensor """
    return x



@augmentation_label_handler
def augment_timeseries_shift(x: tf.Tensor,
                             max_shift: int = 10):
    """ randomly shift the time series """

    # shift the data by removing a random number of later time points
    dt = tf.random.uniform(shape=[], minval=0, maxval=max_shift, dtype=tf.int32)
    return x[:-dt,...]



@augmentation_label_handler
def augment_timeseries_crop(x: tf.Tensor,
                            min_length: int = 30):
    """ randomly remove part of the beginning of the time series """

    max_crop = tf.shape(x)[0] - min_length
    dt = tf.random.uniform(shape=[], minval=0, maxval=max_crop, dtype=tf.int32)
    return x[dt:,...]



@augmentation_label_handler
def augment_timeseries_swap(x: tf.Tensor,
                            n_swaps: int = 10):
    """ randomly swap parts of the time series """

    # get the indices to update
    idx = tf.random.shuffle(tf.range(tf.shape(x)[0]))[:n_swaps]

    # shuffle them and grab the data from the time-series
    shuffled_idx = tf.random.shuffle(idx)
    updates = tf.gather(x, shuffled_idx, axis=0)

    # re-insert the shuffled time-points in the original sequence at idx
    return tf.tensor_scatter_nd_update(x, idx[...,tf.newaxis], updates)



@augmentation_label_handler
def augment_timeseries_shuffle(x: tf.Tensor):
    """ randomly swap all timepoints in the time series """

    # shuffle all time points
    idx = tf.random.shuffle(tf.range(tf.shape(x)[0]))
    return tf.gather(x, idx, axis=0)



@augmentation_label_handler
def augment_timeseries_noise(x: tf.Tensor,
                             stddev_spread: float = 0.15):
    """ add noise to the timeseries """

    # add white noise that scales with the variance of the data
    stddev = stddev_spread * (tf.reduce_max(x) - tf.reduce_min(x))
    noise = tf.random.normal(tf.shape(x), mean=0, stddev=stddev)
    return x + noise



@augmentation_label_handler
def augment_timeseries_corrupt(x: tf.Tensor,
                               n_corrupt: int = 10,
                               stddev_spread: float = 0.15):
    """ corrupt some data from the timeseries by replacing with noise """

    stddev = stddev_spread * (tf.reduce_max(x) - tf.reduce_min(x))
    noise_shape = tf.concat([[n_corrupt], tf.shape(x)[1:]], axis=0)
    noise = tf.random.normal(noise_shape, mean=0, stddev=stddev)

    # get the indices to update
    idx = tf.random.shuffle(tf.range(tf.shape(x)[0]))[:n_corrupt]

    # re-insert the shuffled time-points in the original sequence at idx
    return tf.tensor_scatter_nd_update(x, idx[...,tf.newaxis], noise)



@augmentation_label_handler
def augment_timeseries_dropout(x: tf.Tensor,
                               n_dropout: int = 10):
    """ dropout some observation timepoints, pad from beginning to maintain
    size """

    # get the indices to update
    idx = tf.random.shuffle(tf.range(tf.shape(x)[0]))[:tf.shape(x)[0]-n_dropout]
    dropout = tf.gather(x, idx, axis=0)
    pad_shape = tf.concat([[n_dropout], tf.shape(x)[1:]], axis=0)
    # re-insert the shuffled time-points in the original sequence at idx
    return tf.concat([tf.zeros(pad_shape, dtype=tf.float32), dropout], axis=0)
