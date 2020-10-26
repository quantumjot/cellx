from tensorflow import keras as K


class ConvBlock2D(K.layers.Layer):
    """ ConvBlock2D

    Keras layer to perform a 2D convolution with batch normalization followed
    by activation.

    Parameters
    ----------
    layers : list
        A list of kernels for each layer
    kernel_size : tuple
        Size of the convolutional kernel
    padding : str
        Padding type for convolution
    activation : str
        Name of activation function
    strides : int
        Stride of the convolution

    Notes
    -----
    TODO(arl): accept activation functions as well as names

    """

    def __init__(
        self,
        filters: int = 32,
        kernel_size: tuple = (3, 3),
        padding: str = "same",
        strides: int = 1,
        activation: str = "swish",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.conv = K.layers.Conv2D(
            filters, kernel_size, strides=strides, padding=padding
        )
        self.norm = K.layers.BatchNormalization()
        self.activation = K.layers.Activation(activation)

        self._config = {
            "filters": filters,
            "kernel_size": kernel_size,
            "padding": padding,
            "strides": strides,
            "activation": activation,
        }

    def call(self, x):
        """ return the result of the normalized convolution """
        conv = self.conv(x)
        conv = self.norm(conv)
        return self.activation(conv)

    def get_config(self):
        config = super(ConvBlock2D, self).get_config()
        config.update(self._config)
        return config


class Encoder(K.layers.Layer):
    """Base class for encoders.

    Parameters
    ----------
    layers : list
        A list of kernels for each layer
    kernel_size : tuple
        Size of the convolutional kernel
    padding : str
        Padding type for convolution
    activation : str
        Name of activation function
    use_pooling : bool
        Use pooling or not. If not using pooling, use the stride of the
        convolution to reduce instead.

    Notes
    -----
    The list of kernels can be used to infer the number of conv-pool layers
    in the encoder.
    """

    def __init__(self, layers: list = [8, 16, 32], use_pooling: bool = True):
        pass


class Encoder2D(K.layers.Layer):
    """Encoder2D

    Keras layer to build a stacked encoder using ConvBlock2D.

    Parameters
    ----------
    layers : list
        A list of kernels for each layer
    kernel_size : tuple
        Size of the convolutional kernel
    padding : str
        Padding type for convolution
    activation : str
        Name of activation function
    use_pooling : bool
        Use pooling or not. If not using pooling, use the stride of the
        convolution to reduce instead.

    Notes
    -----
    The list of kernels can be used to infer the number of conv-pool layers
    in the encoder.
    """

    def __init__(
        self,
        layers: list = [8, 16, 32],
        kernel_size: tuple = (3, 3),
        padding: str = "same",
        activation: str = "swish",
        use_pooling: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        if use_pooling:
            self.pool = K.layers.MaxPooling2D()
            strides = 1
        else:
            self.pool = lambda x: x
            strides = 2

        # build the convolutional layer list
        self.layers = [
            ConvBlock2D(
                filters=k,
                kernel_size=kernel_size,
                padding=padding,
                activation=activation,
                strides=strides,
            )
            for k in layers
        ]

        self._config = {
            "layers": layers,
            "kernel_size": kernel_size,
            "padding": padding,
            "activation": activation,
            "use_pooling": use_pooling,
        }

    def call(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.pool(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(self._config)
        return config


class Decoder2D(K.layers.Layer):
    """ Decoder2D

    Keras layer to build a stacked decoder using ConvBlock2D

    Parameters
    ----------
    layers : list
        A list of kernels for each layer
    kernel_size : tuple
        Size of the convolutional kernel
    padding : str
        Padding type for convolution
    activation : str
        Name of activation function

    Notes
    -----
        The list of kernels can be used to infer the number of conv-pool layers
        in the encoder.
    """

    def __init__(
        self,
        layers: list = [8, 16, 32],
        kernel_size: tuple = (3, 3),
        padding: str = "same",
        activation: str = "swish",
        **kwargs
    ):
        super().__init__(**kwargs)

        # build the convolutional layer list
        self.layers = [
            ConvBlock2D(
                filters=k,
                kernel_size=kernel_size,
                padding=padding,
                activation=activation,
            )
            for k in layers
        ]

        self.upsample = K.layers.UpSampling2D()

        self._config = {
            "layers": layers,
            "kernel_size": kernel_size,
            "padding": padding,
            "activation": activation,
        }

    def call(self, x):
        for layer in self.layers:
            x = self.upsample(x)
            x = layer(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(self._config)
        return config


if __name__ == "__main__":
    # boilerplate
    pass
