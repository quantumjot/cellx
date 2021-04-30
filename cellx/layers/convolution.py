from tensorflow import keras as K

from .base import ConvBlockBase, EncoderDecoderBase, ResidualBlockBase


class ConvBlock2D(ConvBlockBase):
    """ConvBlock2D."""

    def __init__(self, convolution=K.layers.Conv2D, **kwargs):
        extra_kwargs = {"convolution": convolution}
        kwargs.update(extra_kwargs)
        super().__init__(**kwargs)


class ConvBlock3D(ConvBlockBase):
    """ConvBlock3D."""

    def __init__(self, convolution=K.layers.Conv3D, **kwargs):
        extra_kwargs = {"convolution": convolution}
        kwargs.update(extra_kwargs)
        super().__init__(**kwargs)


class ResidualBlock2D(ResidualBlockBase):
    """ResidualBlock2D."""

    def __init__(self, convolution=K.layers.Conv2D, **kwargs):
        extra_kwargs = {"convolution": convolution}
        kwargs.update(extra_kwargs)
        super().__init__(**kwargs)


class Encoder2D(EncoderDecoderBase):
    """Encoder2D."""

    def __init__(
        self, convolution=ConvBlock2D, sampling=K.layers.MaxPooling2D, **kwargs
    ):
        extra_kwargs = {"convolution": convolution, "sampling": sampling}
        kwargs.update(extra_kwargs)
        super().__init__(**kwargs)


class Encoder3D(EncoderDecoderBase):
    """Encoder3D."""

    def __init__(
        self, convolution=ConvBlock3D, sampling=K.layers.MaxPooling3D, **kwargs
    ):
        extra_kwargs = {"convolution": convolution, "sampling": sampling}
        kwargs.update(extra_kwargs)
        super().__init__(**kwargs)


class Encoder3DFlat(EncoderDecoderBase):
    """Encoder3DFlat."""

    def __init__(
        self,
        convolution=ConvBlock3D,
        sampling=K.layers.MaxPooling3D(pool_size=(2, 2, 1)),
        **kwargs
    ):
        extra_kwargs = {"convolution": convolution, "sampling": sampling}
        kwargs.update(extra_kwargs)
        super().__init__(**kwargs)


class Decoder2D(EncoderDecoderBase):
    """Decoder2D."""

    def __init__(
        self, convolution=ConvBlock2D, sampling=K.layers.UpSampling2D, **kwargs
    ):
        extra_kwargs = {"convolution": convolution, "sampling": sampling}
        kwargs.update(extra_kwargs)
        super().__init__(**kwargs)


class Decoder3D(EncoderDecoderBase):
    """Decoder3D."""

    def __init__(
        self, convolution=ConvBlock3D, sampling=K.layers.UpSampling3D, **kwargs
    ):
        extra_kwargs = {"convolution": convolution, "sampling": sampling}
        kwargs.update(extra_kwargs)
        super().__init__(**kwargs)


if __name__ == "__main__":
    # boilerplate
    pass
