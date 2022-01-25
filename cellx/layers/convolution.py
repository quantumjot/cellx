from tensorflow import keras as K

from . import base


class ConvBlock2D(base.ConvBlockBase, base.SerializationMixin):
    """ConvBlock2D."""

    def __init__(self, convolution=K.layers.Conv2D, **kwargs):
        extra_kwargs = {"convolution": convolution}
        kwargs.update(extra_kwargs)
        super(base.SerializationMixin, self).__init__(**kwargs)


class ConvBlock3D(base.ConvBlockBase):
    """ConvBlock3D."""

    def __init__(self, convolution=K.layers.Conv3D, **kwargs):
        extra_kwargs = {"convolution": convolution}
        kwargs.update(extra_kwargs)
        super().__init__(**kwargs)


class ResidualBlock2D(base.ResidualBlockBase):
    """ResidualBlock2D."""

    def __init__(self, convolution=K.layers.Conv2D, **kwargs):
        extra_kwargs = {"convolution": convolution}
        kwargs.update(extra_kwargs)
        super().__init__(**kwargs)


class Encoder2D(base.EncoderDecoderBase):
    """Encoder2D."""

    def __init__(
        self, convolution=ConvBlock2D, sampling=K.layers.MaxPooling2D, **kwargs
    ):
        extra_kwargs = {"convolution": convolution, "sampling": sampling}
        kwargs.update(extra_kwargs)
        super().__init__(**kwargs)


class Encoder3D(base.EncoderDecoderBase):
    """Encoder3D."""

    def __init__(
        self, convolution=ConvBlock3D, sampling=K.layers.MaxPooling3D, **kwargs
    ):
        extra_kwargs = {"convolution": convolution, "sampling": sampling}
        kwargs.update(extra_kwargs)
        super().__init__(**kwargs)


class Encoder3DFlat(base.EncoderDecoderBase):
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


class Decoder2D(base.EncoderDecoderBase):
    """Decoder2D."""

    def __init__(
        self, convolution=ConvBlock2D, sampling=K.layers.UpSampling2D, **kwargs
    ):
        extra_kwargs = {"convolution": convolution, "sampling": sampling}
        kwargs.update(extra_kwargs)
        super().__init__(**kwargs)


class Decoder3D(base.EncoderDecoderBase):
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
