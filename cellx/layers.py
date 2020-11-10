from typing import List, Optional, Union

import tensorflow as tf
from tensorflow import keras as K


class ConvBlockBase(K.layers.Layer):
    """Base class for convolutional blocks.

    Keras layer to perform a convolution with batch normalization followed
    by activation.

    Parameters
    ----------
    convolution : keras.layers.Conv
        A convolutional layer for 2 or 3-dimensions
    filters : int
        The number of convolutional filters
    kernel_size : int, tuple
        Size of the convolutional kernel
    padding : str
        Padding type for convolution
    activation : str
        Name of activation function
    strides : int
        Stride of the convolution
    """

    def __init__(
        self,
        convolution: K.layers.Layer = K.layers.Conv2D,
        filters: int = 32,
        kernel_size: Union[int, tuple] = 3,
        padding: str = "same",
        strides: int = 1,
        activation: str = "swish",
        name: str = "ConvBlock",
        **kwargs
    ):
        super().__init__(name=name)

        self.conv = convolution(
            filters, kernel_size, strides=strides, padding=padding, **kwargs,
        )
        self.norm = K.layers.BatchNormalization()
        self.activation = K.layers.Activation(activation)

        # store the config so that we can restore it later
        self._config = {
            "convolution": convolution,
            "filters": filters,
            "kernel_size": kernel_size,
            "padding": padding,
            "strides": strides,
            "activation": activation,
            "name": name,
        }
        self._config.update(kwargs)

    def call(self, x, training: Optional[bool] = None):
        """Return the result of the normalized convolution."""
        conv = self.conv(x)
        conv = self.norm(conv, training=training)
        return self.activation(conv)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(self._config)
        return config


class ConvBlock2D(ConvBlockBase):
    """ConvBlock2D."""

    def __init__(self, **kwargs):
        super().__init__(convolution=K.layers.Conv2D, name="ConvBlock2D", **kwargs)


class ConvBlock3D(ConvBlockBase):
    """ConvBlock3D."""

    def __init__(self, **kwargs):
        super().__init__(convolution=K.layers.Conv3D, name="ConvBlock3D", **kwargs)


class EncoderDecoderBase(K.layers.Layer):
    """Base class for encoders and decoders.

    Parameters
    ----------
    conv_block : ConvBlockBase
        A convolutional block layer
    pooling : K.layers.Layer, None
        If not using pooling, use the stride of the convolution to reduce instead.
    layers : list
        A list of kernels for each layer

    Notes
    -----
    The list of kernels can be used to infer the number of conv-pool layers
    in the encoder.
    """

    def __init__(
        self,
        conv_block: ConvBlockBase = ConvBlock2D,
        pooling: Optional[K.layers.Layer] = K.layers.MaxPooling2D,
        layers: List[int] = [8, 16, 32],
        **kwargs
    ):
        super().__init__()

        if pooling is not None:
            self.pooling = pooling
            strides = 1
        else:
            self.pooling = lambda x: x
            strides = 2

        # build the convolutional layer list
        self.layers = [conv_block(strides=strides, **kwargs) for k in layers]

        self._config = {
            "conv_block": conv_block,
            "layers": layers,
            "pooling": pooling,
        }
        self._config.update(kwargs)

    def call(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.pool(x)
        return x

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(self._config)
        return config


# class Encoder2D(EncoderDecoderBase):
#     def __init__(self, **kwargs):
#         super().__init__(conv_block=ConvBlock2D, **kwargs)


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
        layers: List[int] = [8, 16, 32],
        kernel_size: Union[int, tuple] = 3,
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

    def call(self, x, training: Optional[bool] = None):
        for layer in self.layers:
            x = layer(x, training=training)
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

    def call(self, x, training: Optional[bool] = None):
        for layer in self.layers:
            x = self.upsample(x)
            x = layer(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(self._config)
        return config


class Sampling(K.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = K.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


if __name__ == "__main__":
    # boilerplate
    pass
