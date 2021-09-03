from typing import List, Tuple

import tensorflow.keras as K
from tcn import TCN


def build_split_TCN(
    max_len: int,
    embed_dim: int,
    num_tcn_filters: int = 64,
    num_intermediate_filters: int = 128,
    num_outputs: int = 3,
    kernel_size: int = 2,
    dropout_rate: float = 0.3,
    activation: str = "swish",
    dilations: List[int] = [1, 2, 4, 8, 16, 32, 64],
    kernel_initializer: str = "glorot_uniform",
) -> Tuple[K.Model]:
    """Build predictive TCN with classifier head on final output.

    Parameters
    ----------
    max_len : int
        Maximum frame-length of inputs.
    embed_dim : int
        Size of embedding layer.
    num_tcn_filters : int
        Number of filters in each TCN layer.
    num_intermediate_filters : int
        Number of filters in the intermediate layer of the prediction head.
    num_outputs : int
        Number of outputs.
    kernel_size : int
        Size of kernels in the TCN.
    dropout_rate : float
        Dropout rate used during training of the TCN.
    activation : str
        Name of an activation function for the TCN/Prediction head.
    dilations : list
        A list of dilations for the TCN.
    kernel_initializer : str
        Name of a kernel initializer for the TCN.

    Returns
    -------
    models : tuple (3, )
        Three Keras Models representing:
            * Full_TCN with TCN and prediction head (compiled with Adam optimizer)
            * Prediction_head
            * TCN only
    """
    tcn_layer = TCN(
        nb_filters=num_tcn_filters,
        kernel_size=kernel_size,
        return_sequences=True,
        use_weight_norm=True,
        dropout_rate=dropout_rate,
        activation=activation,
        dilations=dilations,
        kernel_initializer=kernel_initializer,
    )

    # build just the TCN
    i_1 = K.layers.Input(batch_shape=(None, max_len, embed_dim))
    tcn_o = tcn_layer(i_1)
    tcn_split_1 = K.models.Model(inputs=[i_1], outputs=[tcn_o], name="TCN")

    # build the prediction/classification head
    i_2 = K.layers.Input(batch_shape=(None, max_len, num_tcn_filters))
    o = K.layers.Lambda(lambda x: x[:, -1, :])(i_2)
    o = K.layers.Dense(num_intermediate_filters, activation=activation)(o)
    o = K.layers.Dense(num_outputs)(o)
    tcn_split_2 = K.models.Model(inputs=[i_2], outputs=[o], name="Prediction_head")

    # build the full model
    i = K.layers.Input(batch_shape=(None, max_len, embed_dim))
    o_1 = tcn_split_1(i)
    o_2 = tcn_split_2(o_1)
    tcn = K.models.Model(inputs=[i], outputs=[o_2], name="Full_TCN")

    # compile the full TCN
    tcn.compile(
        optimizer=K.optimizers.Adam(learning_rate=0.001),
        loss=K.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return tcn, tcn_split_1, tcn_split_2
