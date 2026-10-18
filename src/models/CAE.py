#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import torch.nn as nn
import torch.nn.functional as nn_func

from math import ceil

from .BaseDetector import BaseDetector, ACTIVATIONS


#============================================================================================================================#
#---------------------------------------------------------- CLASS -----------------------------------------------------------#
#============================================================================================================================#
class ResizeConv1d(nn.Module):
    
    def __init__(self, in_channels, out_channels, scale_factor=2, kernel_size=3, activation : bool = True):

        super(ResizeConv1d, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.conv     = nn.Conv1d(in_channels  = in_channels, 
                                  out_channels = out_channels, 
                                  kernel_size  = kernel_size, 
                                  padding      = kernel_size // 2 
        )

    #------------------------------
    def forward(self, x):

        return self.conv(self.upsample(x))

#================================================================================#
class ConvAutoEncoder(BaseDetector):

    """
    This class define the architecture for a 1D Convolutional 
    Auto-Encodeur CAE, based on BaseDetector general class

    Use 1D convolution to analyse temporal signal form SHM systems

    The network is trained on healthy signals only and detect the gap with 
    a damage signal as the anomaly score computes with the reconstruction MSE.

    Defaults architecture 
    ----------------------
    n_layers      = 3
    base_channels = 16 
    kernel_size   = 7  

    encoder : Conv1d layers with stride = 2 (length divided by 2**n_layers) 
            -> channels : 1 
            -> channels : 16 (kernel_size = 7) 
            -> channels : 32 (kernel_size = 5) 
            -> channels : 64 (kernel_size = 3)

    decoder : symetric with the encoder
              -> 'resize'    : Upsample + Conv1D
              -> 'transpose' : ConvTranspose1D

    """

    #---------------------------------------------
    NAME = "CAE"

    DEFAULTS = {

        #--------------------
        "n_layers"        : 5,
        "latent_channels" : 4,
        "base_channels"   : 16,
        "kernel_size"     : 7,
        "activation"      : "relu",
        "decode_method"   : "resize",

        #--------------------
        "lr"            : 1e-3,
        "batch_size"    : 32,
        "optimizer"     : "adam",
        "weight_decay"  : 0.0,
    }

    SEARCH_SPACE = {
        "n_layers"        : [4, 5, 6],
        "latent_channels" : [2, 4, 8],
        "base_channels"   : [8, 16, 32],
        "kernel_size"     : [3, 5, 7, 9],

        "activation"      : ["relu", "leaky_relu", "gelu"],
        "decode_method"   : ["resize", "transpose"],

        "lr"              : [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        "batch_size"      : [16, 32, 64],
    }

    #================================================================================#
    def __init__(self,
                 input_length    : int,
                 in_channels     : int = 1,
                 n_layers        : int = 3,
                 base_channels   : int = 16,
                 kernel_size     : int = 7,
                 latent_channels : int = 4,
                 activation      : str = "relu",
                 decode_method   : str = "resize"):

        """
        Args
        ----------
        input_length  : (int) input signal length
        in_channels   : (int) 

        n_layers      : (int) number of Conv1D layer in the network (stride = 2 (length divided by 2**n_layers))

        latent_channels : (int) channels of the bottleneck (1x1 conv after the encoder) -- defaults latent_channels = 4
                          latent size = latent_channels x input_length / 2**n_layers    -> must be < input_length

        base_channels : (int) number channels of the first layer, doubled at each layer -- defaults base_channels = 16
        kernel_size   : (int) kernel size of the first layer, reduced by 2 at each layer -- defaults kernel_size = 7

        activation    : (str) activation of every hidden layer, key of ACTIVATIONS
                        ('relu', 'leaky_relu', 'gelu', 'elu') -- defaults activation = 'relu'

        decode_method : (str) string to set the decoder method -- defaults decode_method = 'resize'
                        'resize' for Resize-Conv | 'transpose' for ConvTranspose

        Examples
        ----------
            - ConvAutoEncoder(input_length = 2000, n_layers = 4, base_channels = 16, kernel_size = 7) 

            - ConvAutoEncoder(input_length=2000, decode_method="resize")
        """

        #---------------------------------------------
        super(ConvAutoEncoder, self).__init__(input_length, in_channels=in_channels)

        if(decode_method not in ("resize", "transpose")):
            raise ValueError(f"decode_method must be 'transpose' or 'resize' : {decode_method}")

        if(activation not in ACTIVATIONS):
            raise ValueError(f"activation must be one of {list(ACTIVATIONS)} : {activation}")

        latent_size = latent_channels * ceil(input_length / 2 ** n_layers)

        if(latent_size >= input_length * in_channels):
            raise ValueError(f"no bottleneck : latent {latent_channels}x{ceil(input_length / 2 ** n_layers)} "
                            f"= {latent_size} >= input {input_length} (increase n_layers or decrease latent_channels)")
        
        #---------------------------------------------
        channels = [in_channels] + [base_channels * 2 ** i for i in range(n_layers)]
        kernels  = [max(3, kernel_size - 2 * i) | 1 for i in range(n_layers)]

        #---------------------------------------------
        encoder = []
        for i in range(n_layers):

            conv_i  = nn.Conv1d(in_channels   = channels[i], 
                                out_channels  = channels[i + 1],
                                kernel_size   = kernels[i], 
                                stride        = 2, 
                                padding       = kernels[i] // 2)

            activ_i = ACTIVATIONS[activation]()

            encoder.extend([conv_i, activ_i])

        #---------------
        bottleneck   = nn.Conv1d(channels[-1], latent_channels, kernel_size=1)
        self.encoder = nn.Sequential(*encoder, bottleneck)

        #---------------------------------------------
        decoder = []
        for i in reversed(range(n_layers)):

            k        = kernels[i]
            chan_in  = channels[i + 1]
            chan_out = channels[i]

            #---------------
            if decode_method == "resize":
                layer_i = ResizeConv1d(in_channels  = chan_in,
                                       out_channels = chan_out,
                                       scale_factor = 2,
                                       kernel_size  = k)
            else:
                layer_i = nn.ConvTranspose1d(in_channels    = chan_in,
                                             out_channels   = chan_out,
                                             kernel_size    = k,
                                             stride         = 2,
                                             padding        = k // 2,
                                             output_padding = 1)

            activ_i = ACTIVATIONS[activation]() if i != 0 else nn.Identity()

            decoder.extend([layer_i, activ_i])

        #---------------
        bottleneck_up = nn.Conv1d(latent_channels, channels[-1], kernel_size=1)

        self.decoder = nn.Sequential(bottleneck_up, ACTIVATIONS[activation](), *decoder)

    #================================================================================#
    def forward(self, x):

        """
        process the input signal through the network and return its reconstruction
        """

        #---------------------------------------------
        z     = self.encoder(x)
        recon = self.decoder(z)

        return recon[..., : x.shape[-1]]

    #================================================================================#
    def training_step(self, batch):

        x = batch[0]
        return nn_func.mse_loss(self(x), x)

    #================================================================================#
    def anomaly_score(self, x):

        mse = ((self(x) - x) ** 2).mean(dim=2).amax(dim=1)
        return (mse)