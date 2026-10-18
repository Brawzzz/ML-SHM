#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import torch.nn as nn
import torch.nn.functional as nn_func

from .BaseDetector import BaseDetector, ACTIVATIONS


#============================================================================================================================#
#---------------------------------------------------------- CLASS -----------------------------------------------------------#
#============================================================================================================================#
class AutoEncoder(BaseDetector):

    """
    Dense Auto-Encoder - AE, based on BaseDetector general class

    The whole signal is flattened and compressed into a latent vector of `latent_dim` values
    by a stack of Linear layers, then reconstructed by a symmetric decoder.

    The network is trained on healthy signals only. The anomaly score is the reconstruction MSE.

    Defaults architecture  (input_length = 2000)
    ----------------------
    n_layers     = 3
    first_hidden = 512
    latent_dim   = 32

    encoder : 2000 -> 512 -> 256 -> 128 -> 32     (Linear + activation [+ dropout], last one linear)
    decoder :   32 -> 128 -> 256 -> 512 -> 2000   (symmetric, output layer linear)
    """

    #---------------------------------------------
    NAME = "AE"

    DEFAULTS = {

        #--------------------
        "n_layers"     : 3,
        "first_hidden" : 512,
        "latent_dim"   : 32,
        "activation"   : "relu",
        "dropout"      : 0.0,

        #--------------------
        "lr"           : 1e-3,
        "batch_size"   : 64,
        "optimizer"    : "adam",
        "weight_decay" : 0.0,
    }

    SEARCH_SPACE = {
        "n_layers"     : [2, 3, 4],
        "first_hidden" : [512, 1024],                      
        "latent_dim"   : [8, 16, 32],                       
        "activation"   : ["relu", "leaky_relu", "gelu"],
        "dropout"      : [0.0, 0.1, 0.2],

        "lr"           : [1e-4, 3e-4, 1e-3, 3e-3],
        "batch_size"   : [32, 64, 128],
    }

    #================================================================================#
    def __init__(self,
                 input_length : int,
                 in_channels  : int = 1,
                 n_layers     : int   = 3,
                 first_hidden : int   = 512,
                 latent_dim   : int   = 32,
                 activation   : str   = "relu",
                 dropout      : float = 0.0):

        """
        Args
        ----------
        input_length : (int) input signal length

        n_layers     : (int) number of hidden layers in the encoder (the decoder is symmetric)
        first_hidden : (int) width of the first hidden layer, halved at each layer -- defaults first_hidden = 512
        latent_dim   : (int) size of the latent vector (bottleneck) -- defaults latent_dim = 32
                       must be smaller than every hidden layer and than input_length

        activation   : (str) activation of every hidden layer, key of ACTIVATIONS
                       ('relu', 'leaky_relu', 'gelu', 'elu') -- defaults activation = 'relu'
        dropout      : (float) dropout probability after each hidden activation -- defaults dropout = 0.0

        Examples
        ----------
            - AutoEncoder(input_length = 2000)

            - AutoEncoder(input_length = 2000, n_layers = 2, first_hidden = 256, latent_dim = 16)
        """

        #---------------------------------------------
        super(AutoEncoder, self).__init__(input_length, in_channels)

        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {list(ACTIVATIONS)} : {activation}")

        hidden = [first_hidden // 2 ** i for i in range(n_layers)]

        n_in  = in_channels * input_length

        if latent_dim >= min(hidden) or latent_dim >= n_in:
            raise ValueError(f"no bottleneck : latent_dim = {latent_dim} must be < every hidden layer {hidden} "
                             f"and < input_length = {input_length}")

        #---------------------------------------------
        sizes = [n_in] + hidden + [latent_dim]                     

        self.encoder = _mlp(sizes, activation, dropout)
        self.decoder = _mlp(sizes[::-1], activation, dropout)               

    #================================================================================#
    def forward(self, x):

        """
        x : (batch, 1, input_length) -> reconstruction of the same shape
        """

        #---------------------------------------------
        z     = self.encoder(x.flatten(start_dim=1))                           # (batch, latent_dim)
        recon = self.decoder(z)                                                # (batch, input_length)

        return recon.view_as(x)

    #================================================================================#
    def training_step(self, batch):

        x = batch[0]
        return nn_func.mse_loss(self(x), x)

    #================================================================================#
    def anomaly_score(self, x):

        mse = ((self(x) - x) ** 2).mean(dim=(1, 2))
        return mse

#============================================================================================================================#
#-------------------------------------------------------- FUNCTIONS ---------------------------------------------------------#
#============================================================================================================================#
def _mlp(sizes: list[int], activation: str, dropout: float) -> nn.Sequential:

    """
    Stack of Linear layers following `sizes`.
    Every layer is followed by activation (+ dropout) except the LAST one, kept linear
    (linear latent for the encoder, linear output for the decoder).

    Args
    ----------
    sizes      : (list[int]) layer widths, e.g. [2000, 512, 256, 128, 32]
    activation : (str) key of ACTIVATIONS
    dropout    : (float) dropout probability -- 0 = no dropout layer

    Returns
    ----------
    (nn.Sequential) the stacked layers
    """

    #---------------------------------------------
    layers = []
    for i in range(len(sizes) - 1):

        layers.append(nn.Linear(sizes[i], sizes[i + 1]))

        if i < len(sizes) - 2:
            layers.append(ACTIVATIONS[activation]())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)