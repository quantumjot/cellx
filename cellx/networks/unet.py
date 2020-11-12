from enum import Enum
from typing import List, Optional

from tensorflow import keras as K

from ..layers import ConvBlock2D


class SkipConnection(Enum):
    """Skip connections for UNet."""

    ELEMENTWISE_ADD = K.layers.Add()
    ELEMENTWISE_MULTIPLY = K.layers.Multiply()
    CONCATENATE = K.layers.Concatenate(axis=-1)
    NONE = lambda x: x[-1]

    def __call__(self, inputs):
        return self.value(inputs)


class UNet(K.Model):
    """ UNet

    A UNet class for image segmentation. This implementation differs in that we
    pad each convolution such that the output following convolution is the same
    size as the input. Also, bridges are elementwise operations of the filters
    to approach a residual-net architecture (resnet), although this can be
    changed by the user.  The skip property allows different skip connection
    types to be specified:
        - elementwise_add
        - elementwise_multiply
        - elementwise_subtract
        - concatenate
        - None (no bridge information, resembles an autoencoder)

    Image autoencoders can also be subclassed from this structure, by
    removing the bridge information.

    Note that the UNet class should not be used on it's own. Generally
    there are subclassed versions which inherit the main features but
    specify loss functions and bridge details that are specific to the
    particular architecture.

    ** The final layer does not have an activation function. **

    Parameters
    ----------
    convolution : K.layers.Layer
    downsampling : K.layers.Layer
    upsampling : K.layers.Layer
    layers : list of ints
    skip : str
        The skip connection type.
    outputs : int
        The number of output channels.
    name : str
        The name of the network.

    Notes
    -----
    Based on the original publications:

    U-Net: Convolutional Networks for Biomedical Image Segmentation
    Olaf Ronneberger, Philipp Fischer and Thomas Brox
    http://arxiv.org/abs/1505.04597

    3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation
    Ozgun Cicek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox
    and Olaf Ronneberger
    https://arxiv.org/abs/1606.06650

    Filter doubling from:
    Rethinking the Inception Architecture for Computer Vision.
    Szegedy C., Vanhoucke V., Ioffe S., Shlens J., Wojn, Z.
    https://arxiv.org/abs/1512.00567
    """

    def __init__(
        self,
        convolution: K.layers.Layer = ConvBlock2D,
        downsampling: K.layers.Layer = K.layers.MaxPooling2D(pool_size=(2, 2)),
        upsampling: K.layers.Layer = K.layers.UpSampling2D(size=(2, 2)),
        layers: List[int] = [8, 16, 32],
        outputs: int = 1,
        skip: str = "concatenate",
        name: str = "unet",
        **kwargs,
    ):

        super().__init__(name=name, **kwargs)

        # convert the type here
        if skip.upper() not in SkipConnection._member_names_:
            raise ValueError(f"Skip connection {skip} not recognized.")
        self.skip = SkipConnection[skip.upper()]

        self.convolution = convolution
        self.downsampling = downsampling
        self.upsampling = upsampling

    def build(self, input_shape: tuple):
        """Build the UNet"""
        pass

    def call(self, x, training: Optional[bool] = None):
        pass
