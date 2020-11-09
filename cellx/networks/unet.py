from enum import Enum

from tensorflow import keras as K

from ..layers import Decoder2D, Encoder2D


class SkipConnection(Enum):
    """Skip connections for UNet."""

    ELEMENTWISE_ADD = K.layers.Add
    ELEMENTWISE_MULTIPLY = K.layers.Multiply
    CONCATENATE = K.layers.Concatenate
    NONE = lambda x: x[-1]


class UNetBase(K.Model):
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
    encoder : cellx.layers.Encoder, None
        An encoder layer.
    decoder : cellx.layers.Decoder, None
        A decoder layer.
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
        encoder: K.layers.Layer = Encoder2D,
        decoder: K.layers.Layer = Decoder2D,
        outputs: int = 1,
        skip: str = "concatenate",
        name: str = "unet",
        **kwargs,
    ):

        super().__init__(name=name, **kwargs)
        self.encoder = encoder
        self.decoder = decoder

        if encoder not in (Encoder2D,):
            raise ValueError(f"Encoder {encoder} not recognized.")

        if decoder not in (Decoder2D,):
            raise ValueError(f"Decoder {decoder} not recognized.")

        # convert the type here
        if isinstance(skip, str):
            self.skip = SkipConnection[skip.uppercase()]
        else:
            self.skip = skip

    def build(self, input_shape):
        pass

    def call(self, inputs):
        pass
