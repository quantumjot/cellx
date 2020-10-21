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
                 kernel_size: tuple = (3, 3),
                 padding: str = 'same',
                 activation: str = 'swish',
                 **kwargs):
        super(ConvBlock2D, self).__init__(**kwargs)

        self.conv = K.layers.Conv2D(filters, kernel_size, padding=padding)
        self.norm = K.layers.BatchNormalization()
        self.activation = K.layers.Activation(activation)

        self._config = {'filters': filters,
                        'kernel_size': kernel_size,
                        'padding': padding,
                        'activation': activation}

    def call(self, x):
        """ return the result of the normalized convolution """
        conv = self.conv(x)
        conv = self.norm(conv)
        return self.activation(conv)

    def get_config(self):
        config = super(ConvBlock2D, self).get_config()
        config.update(self._config)
        return config


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
        self.layers = [ConvBlock2D(filters=k,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   activation=activation) for k in layers]

        self.pool = K.layers.MaxPooling2D()

        self._config = {'layers': layers,
                        'kernel_size': kernel_size,
                        'padding': padding,
                        'activation': activation}

    def call(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.pool(x)
        return x

    def get_config(self):
        config = super(Encoder2D, self).get_config()
        config.update(self._config)
        return config


class Decoder2D(K.layers.Layer):
    """ Decoder2D

    Keras layer to build a stacked decoder using ConvBlock2D

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
        super(Decoder2D, self).__init__(**kwargs)

        # build the convolutional layer list
        self.layers = [ConvBlock2D(filters=k,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   activation=activation) for k in layers]

        self.upsample = K.layers.UpSampling2D()

        self._config = {'layers': layers,
                        'kernel_size': kernel_size,
                        'padding': padding,
                        'activation': activation}

    def call(self, x):
        for layer in self.layers:
            x = self.upsample(x)
            x = layer(x)
        return x

    def get_config(self):
        config = super(Decoder2D, self).get_config()
        config.update(self._config)
        return config


if __name__ == '__main__':
    # boilerplate
    pass
