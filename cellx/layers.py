import numpy as np
from tensorflow import keras as K

class ConvBlock2D(K.layers.Layer):
    """ ConvBlock2D

    Keras layer to perform a 2D convolution with batch normalization followed
    by activation

    Params:
        filters: int, number of kernels for the 2D convolution
        kernel_size: tuple,
        padding: str,
        activation: str, name of activation function

    Notes:
        TODO(arl): accept activation functions as well as names

    """
    def __init(self,
               filters: int = 32,
               kernel_size: tuple = (3, 3),
               padding: str = 'same',
               activation: str = 'swish',
               **kwargs):
        super(ConvBlock2D, self).__init__(**kwargs)

        self.conv = K.layers.Conv2D(filters, kernel_size, padding=padding)
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
        layers: list, a list of kernels for each layer
        kernel_size: tuple,
        padding: str,
        activation: str, name of activation function

    Notes:
        The list of kernels can be used to infer the number of conv-pool layers
        in the encoder.
    """
    def __init__(self,
                 layers: list = [8, 16, 32],
                 kernel_size: tuple = (3, 3),
                 padding: str = 'same',
                 activation: str = 'swish',
                 **kwargs):
        super(Encoder2D, self).__init__(**kwargs)

        # build the convolutional layer list
        # self.layers = [ConvBlock2D(l, kernel_size) for l in layers]

    def build(self, input_size):
        """ build the encoder network """

    def call(self, x):
        pass



if __name__ == '__main__':
    # boilerplate
    pass
