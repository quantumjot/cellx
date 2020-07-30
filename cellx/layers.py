import numpy as np
from tensorflow import keras as K

class ConvBlock2D(K.layers.Layer):
    """ ConvBlock2D

    Keras layer to perform a 2D convolution with batch normalization followed
    by activation

    Params:
        units: int, number of kernels for the 2D convolution
        activation: str, name of activation function

    Notes:
        TODO(arl): accept activation functions as well as names

    """
    def __init(self,
               units: int = 32,
               activation: str = 'swish',
               **kwargs):
        super(ConvBlock2D, self).__init__(**kwargs)

        self.conv = K.layers.Conv2D(units, (3, 3), padding='same')
        self.norm = K.layers.BatchNormalization()
        self.activation = K.layers.Activation(activation)

    def call(self, x):
        """ return the result of the normalized convolution """
        conv = self.conv(x)
        conv = self.norm(conv)
        return self.activation(conv)





class Encoder2D(K.layers.Layer):
    """ Encoder2D

    Keras layer to build a stacked encoder using ConvBlock2D

    Params:
        kernels: list, a list of kernels for each layer
        activation: str, name of activation function

    Notes:
        The list of kernels can be used to infer the number of conv-pool layers
        in the encoder.
    """
    def __init__(self,
                 kernels: list = [8, 16, 32],
                 activation: str = 'swish',
                 **kwargs):
        super(Encoder2D, self).__init__(**kwargs)

    def call(self, x):
        pass



if __name__ == '__main__':
    # boilerplate
    pass
