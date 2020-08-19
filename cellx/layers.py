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
    def __init__(self,
                 filters: int = 32,
                 kernel_size: tuple = (3,3),
                 padding: str = 'same',
                 activation: str = 'swish'):
        super(ConvBlock2D, self).__init__()

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
                 activation: str = 'swish'):
        super(Encoder2D, self).__init__()

        # build the convolutional layer list
        self.layers = [ConvBlock2D(filters=k,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   activation=activation) for k in layers]

        self.pool = K.layers.MaxPooling2D()

    def call(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.pool(x)
        return x



if __name__ == '__main__':
    # boilerplate
    pass
