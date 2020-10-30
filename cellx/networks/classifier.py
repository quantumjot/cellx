# Building the CNN classifier:

import tensorflow as tf
from tensorflow import keras as K

from cellx import layers

input = K.Input(shape=(2,))
encoder = layers.Encoder2D()(input)
dense = K.layers.Dense(512, activation="relu")(encoder)
output = K.layers.Dense(4, activation=tf.nn.softmax)(dense)

classifier_model = K.Model(inputs=input, outputs=output)
