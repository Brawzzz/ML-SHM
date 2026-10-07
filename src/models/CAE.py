#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .base import BaseDetector


#============================================================================================================================#
#---------------------------------------------------------- CLASS -----------------------------------------------------------#
#============================================================================================================================#
class ResizeConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor=2, kernel_size=3, activation=True):

        super(ResizeConv1d, self).__init__()

        self.upsample   = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.conv       = nn.Conv1d(
            in_channels     = in_channels,
            out_channels    = out_channels,
            kernel_size     = kernel_size,
            padding         = kernel_size // 2
        )

        self.activation = nn.ReLU() if activation else nn.Identity()

    #------------------------------
    def forward(self, x):

        x = self.upsample(x)
        x = self.conv(x)
        x = self.activation(x)

        return x

#================================================================================#
class ConvAutoEncoder(BaseDetector):

    """
    Convolutional Auto-Encoder (CAE) on 1D temporal signals from SHM systems.

    Trained on healthy signals only, the anomaly score is the reconstruction MSE.

    Architecture (n_layers = 3, base_channels = 16, kernel_size = 7 -> previous fixed model) :
        encoder : Conv1d stride 2 : 1 -> 16 (k7) -> 32 (k5) -> 64 (k3)
        decoder : mirrored, by 'resize' (Upsample + Conv) or 'transpose' (ConvTranspose)

    Any signal length is accepted : the reconstruction is cropped to the input length.
    """

    NAME = "CAE"

    #---------------------------------------------
    DEFAULTS = {
        # architecture
        "n_layers"      : 3,
        "base_channels" : 16,
        "kernel_size"   : 7,
        "decode_method" : "resize",
        # optimisation
        "lr"            : 1e-3,
        "batch_size"    : 32,
        "optimizer"     : "adam",
        "weight_decay"  : 0.0,
    }

    #---------------------------------------------
    SEARCH_SPACE = {
        "n_layers"      : {"type": "int",         "low": 2,    "high": 4},
        "base_channels" : {"type": "categorical", "choices": [8, 16, 32]},
        "kernel_size"   : {"type": "categorical", "choices": [3, 5, 7, 9]},
        "decode_method" : {"type": "categorical", "choices": ["resize", "transpose"]},
        "lr"            : {"type": "float",       "low": 1e-4, "high": 1e-2, "log": True},
        "batch_size"    : {"type": "categorical", "choices": [16, 32, 64]},
    }

    #================================================================================#
    def __init__(self,
                 input_length: int,
                 n_layers: int = 3,
                 base_channels: int = 16,
                 kernel_size: int = 7,
                 decode_method: str = "resize"):

        """
        Parameters
        ----------
        input_length  : input signal length
        n_layers      : number of stride-2 convolutions in the encoder (length divided by 2**n_layers)
        base_channels : channels of the first layer, doubled at each layer
        kernel_size   : kernel of the first layer, reduced by 2 at each layer (min 3)
        decode_method : 'resize' for Resize-Conv / 'transpose' for ConvTranspose
        """

        super(ConvAutoEncoder, self).__init__(input_length)

        if decode_method not in ("resize", "transpose"):
            raise ValueError(f"decode_method must be 'transpose' or 'resize' : {decode_method}")

        #---------------------------------------------
        channels = [1] + [base_channels * 2 ** i for i in range(n_layers)]
        kernels  = [max(3, kernel_size - 2 * i) | 1 for i in range(n_layers)]     # odd kernels

        #---------------------------------------------
        enc = []
        for i in range(n_layers):
            enc += [nn.Conv1d(channels[i], channels[i + 1], kernel_size=kernels[i], stride=2, padding=kernels[i] // 2),
                    nn.ReLU()]

        self.encoder = nn.Sequential(*enc)

        #---------------------------------------------
        dec = []
        for i in reversed(range(n_layers)):

            last = (i == 0)                                        # last layer -> 1 channel, no activation
            (c_in, c_out, k) = (channels[i + 1], channels[i], kernels[i])

            if decode_method == "transpose":
                dec.append(nn.ConvTranspose1d(c_in, c_out, kernel_size=k, stride=2, padding=k // 2, output_padding=1))
                if not last:
                    dec.append(nn.ReLU())

            else:
                dec.append(ResizeConv1d(c_in, c_out, scale_factor=2, kernel_size=k, activation=not last))

        self.decoder = nn.Sequential(*dec)

    #================================================================================#
    def forward(self, x):

        """
        process the input signal (batch, 1, length) through the network and return its reconstruction
        """

        #---------------------------------------------
        z     = self.encoder(x)
        recon = self.decoder(z)

        # decoder output length = 2**n_layers * ceil(L / 2**n_layers) >= L -> crop
        return recon[..., : x.shape[-1]]

    #================================================================================#
    def training_step(self, batch):

        x = batch[0]
        return F.mse_loss(self(x), x)

    #================================================================================#
    def anomaly_score(self, x):

        return ((self(x) - x) ** 2).mean(dim=(1, 2))


#============================================================================================================================#
#--------------------------------------------------------- FUNCTION ---------------------------------------------------------#
#============================================================================================================================#
def CAE_plot(X_crack: np.ndarray, crack_recon: np.ndarray, warning_threshold: float, index_to_plot: int = 0) -> None:

    """
    Plot a graph comparing the original cracks signal and its reconstruction by the autoencoder.
    Colour the areas where the reconstruction error exceeds the critical threshold.

    Parameters
    ----------
    X_crack             : NumPy array containing des originals crack signals
    crack_recon         : NumPy array containing reconstruct signals from AE.
    warning_threshold   : MSE threshold for an anomaly
    index_to_plot       : index of the signal to plot
    """

    #---------------------------------------------
    if len(X_crack) == 0 or index_to_plot >= len(X_crack):
        print(f"no signals fouds len(X_crack) = {len(X_crack)} OR index out of range : {index_to_plot}.")
        return

    signal_size = X_crack.shape[1]

    plt.figure(figsize=(14, 5))

    plt.plot(X_crack[index_to_plot], label='Signal original (Fissuré)', color='red', alpha=0.8, linewidth=1.5)
    plt.plot(crack_recon[index_to_plot], label="Signal reconstruit ", color='blue', linestyle='--', linewidth=1.5)

    error_abs           = np.abs(X_crack[index_to_plot] - crack_recon[index_to_plot])
    visual_threshold    = 2.5 * np.sqrt(warning_threshold)

    plt.fill_between(
        range(signal_size),
        -2.0,
        2.5,
        where=(error_abs > visual_threshold),
        color='orange',
        alpha=0.3,
        zorder=0,
        label="Zone d'Anomalie Détectée"
    )

    #------------------------------
    plt.title(f"Analyse de l'Anomalie - Échec de reconstruction (Index {index_to_plot})", fontsize=14, fontweight='bold')
    plt.xlabel("Échantillons temporels", fontsize=12)
    plt.ylabel("Amplitude (Normalisée)", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()

    plt.show()
