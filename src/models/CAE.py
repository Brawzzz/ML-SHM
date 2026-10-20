#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import torch
import torch.nn as nn
import torch.nn.functional as nn_func

from .BaseDetector import BaseDetector, ACTIVATIONS


#============================================================================================================================#
#---------------------------------------------------------- CLASS -----------------------------------------------------------#
#============================================================================================================================#
class ResizeConv1d(nn.Module):

    """
    Decoder block : Upsample (nearest, x scale_factor) then Conv1d with 'same' padding
    """

    def __init__(self, in_channels, out_channels, scale_factor=2, kernel_size=3):

        super(ResizeConv1d, self).__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.conv     = nn.Conv1d(in_channels  = in_channels,
                                  out_channels = out_channels,
                                  kernel_size  = kernel_size,
                                  padding      = "same")

    #------------------------------
    def forward(self, x):

        return self.conv(self.upsample(x))

#================================================================================#
class ConvAutoEncoder(BaseDetector):

    """
    1D Convolutional Auto-Encoder (CAE), based on BaseDetector general class.

    The network is trained on healthy signals only. The anomaly score is the reconstruction MSE
    (mean over time, max over channels).

    Every architectural choice is a hyperparameter, so it can be set from the config
    (model.params) or tuned (model.search_space) :

    Attributes
    ----------
    input_length : (int) length L of the input signals
    in_channels  : (int) number of input channels C (actuator-receiver paths)

    latent_dim    : (int | None) size of the dense code -- None = convolutional bottleneck (1x1 conv)
    encoder_shape : (tuple) shape at the output of the convolutional encoder -- (C_enc, L_enc)
                    e.g. Sawant config, L = 5000 : (16, 1250)

    encoder      : (nn.Sequential) conv blocks, each one divides the length by 2
                    (+ 1x1 conv to latent_channels if latent_dim is None)
    decoder      : (nn.Sequential) mirror of the encoder, output of shape (C, L)

    to_latent    : (nn.Sequential) Flatten -> Linear(C_enc * L_enc, latent_dim)    -- only if latent_dim
    from_latent  : (nn.Sequential) Linear(latent_dim, C_enc * L_enc) -> activation -- only if latent_dim
    
    Encoder
    ----------
        downsample = "stride"  : Conv1d(stride=2) -> activation
        downsample = "maxpool" : Conv1d(stride=1, 'same') -> activation -> MaxPool1d(2)

        filters  : channels[i] if `channels` is given, else base_channels * 2**i
        kernels  : kernels[i]  if `kernels`  is given, else max(3, kernel_size - 2*i) (odd)

    Bottleneck
    ----------
        latent_dim = None : Conv1d 1x1 -> latent_channels x (L / 2**n_layers)
        latent_dim = d    : Flatten -> Dense(d) (linear code) -> Dense -> activation -> Reshape

    Decoder 
    ----------
        mirror with the encoder

        decode_method = "resize"    : Upsample + Conv1d
        decode_method = "transpose" : ConvTranspose1d (only with downsample = "stride")

    Examples
    ----------
        default (unchanged behaviour) : {}

        Sawant et al. 2023 (Ultrasonics 130) : 
        Conv(16, k=8) -> MaxPool -> Conv(16, k=16) -> MaxPool ->Dense(16) :
        
            config : {
                "n_layers"  : 2, 
                "channels": [16, 16], 
                "kernels": [8, 16], 
                "downsample": 
                "maxpool", 
                "latent_dim": 16, 
                "decode_method": "resize"
            }
    """

    #---------------------------------------------
    NAME = "CAE"

    DEFAULTS = {

        #-------------------- ENDODER
        "n_layers"        : 5,
        "base_channels"   : 16,
        "kernel_size"     : 7,
        "channels"        : None,           # explicit list of filters per layer -- overrides base_channels
        "kernels"         : None,           # explicit list of kernel sizes per layer -- overrides kernel_size
        "downsample"      : "stride",       # "stride" | "maxpool"
        "activation"      : "relu",

        #-------------------- BOTTLENECK
        "latent_channels" : 4,              
        "latent_dim"      : None,           # int -> dense bottleneck of given size

        #-------------------- DECODER
        "decode_method"   : "resize",       # "resize" | "transpose"

        #-------------------- TRAINING
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

    input_length  : int                     
    in_channels   : int                     

    latent_dim    : int | None              # size of the dense code -- None = convolutional bottleneck
    encoder_shape : tuple[int, int]         # (C_enc, L_enc) shape at the output of self.encoder

    encoder       : nn.Sequential 
    decoder       : nn.Sequential

    to_latent     : nn.Sequential           # only if latent_dim : Flatten -> Linear(C_enc * L_enc, latent_dim)
    from_latent   : nn.Sequential           # only if latent_dim : Linear(latent_dim, C_enc * L_enc) -> activation
    
    #================================================================================#
    def __init__(self,
                 input_length    : int,
                 in_channels     : int        = 1,
                 n_layers        : int        = 5,
                 base_channels   : int        = 16,
                 kernel_size     : int        = 7,
                 channels        : list[int]  = None,
                 kernels         : list[int]  = None,
                 downsample      : str        = "stride",
                 activation      : str        = "relu",
                 latent_channels : int        = 4,
                 latent_dim      : int        = None,
                 decode_method   : str        = "resize"):

        """
        Args
        ----------
        input_length    : (int) input signal length
        in_channels     : (int) number of input channels (actuator-receiver paths) -- defaults 1

        n_layers        : (int) number of encoder blocks, each one divides the length by 2
        base_channels   : (int) filters of the first layer, doubled at each layer (ignored if `channels` is given)
        kernel_size     : (int) kernel of the first layer, reduced by 2 at each layer (ignored if `kernels` is given)
        channels        : (list[int]) filters of each encoder layer -- len must be n_layers
        kernels         : (list[int]) kernel size of each encoder layer -- len must be n_layers
                          (even sizes are allowed with downsample = "maxpool")

        downsample      : (str) "stride" (strided conv) | "maxpool" (conv stride 1 + MaxPool1d(2))
        activation      : (str) key of ACTIVATIONS ('relu', 'leaky_relu', 'gelu', 'elu')

        latent_channels : (int) channels of the 1x1 conv bottleneck (when latent_dim is None)
        latent_dim      : (int) size of a dense code -- None = convolutional bottleneck
                          the latent size must be < in_channels x input_length

        decode_method   : (str) 'resize' (Upsample + Conv1d) | 'transpose' (ConvTranspose1d)
        """

        #---------------------------------------------
        super(ConvAutoEncoder, self).__init__(input_length, in_channels=in_channels)

        if decode_method not in ("resize", "transpose"):
            raise ValueError(f"decode_method must be 'transpose' or 'resize' : {decode_method}")

        if downsample not in ("stride", "maxpool"):
            raise ValueError(f"downsample must be 'stride' or 'maxpool' : {downsample}")

        if activation not in ACTIVATIONS:
                    raise ValueError(f"activation must be one of {list(ACTIVATIONS)} : {activation}")

        if (downsample == "maxpool") and (decode_method == "transpose"):
            raise ValueError("decode_method 'transpose' is only available with downsample 'stride'")

        #---------------------------------------------
        if channels is not None :
            filters = list(channels)  
        else :
            filters = [base_channels * 2 ** i for i in range(n_layers)]

        #--------------------
        if kernels  is not None : 
            k_sizes = list(kernels)  
        else : 
            k_sizes = [max(3, kernel_size - 2 * i) for i in range(n_layers)]

        #--------------------
        if (len(filters) != n_layers) or (len(k_sizes) != n_layers):
            raise ValueError(f"channels ({len(filters)}) and kernels ({len(k_sizes)}) must have n_layers = {n_layers} values")

        #--------------------
        if downsample == "stride":
            k_sizes = [k | 1 for k in k_sizes]                  

        chans = [in_channels] + filters

        #--------------------------------------------- ENCODER
        encoder = []
        for i in range(n_layers):

            if downsample == "stride":
                encoder += [nn.Conv1d(chans[i], chans[i + 1], k_sizes[i], stride=2, padding=k_sizes[i] // 2),
                            ACTIVATIONS[activation]()]
                
            else:
                encoder += [nn.Conv1d(chans[i], chans[i + 1], k_sizes[i], padding="same"),
                            ACTIVATIONS[activation](),
                            nn.MaxPool1d(2, ceil_mode=True)]

        #---------------
        self.latent_dim = latent_dim

        if latent_dim is None:
            encoder.append(nn.Conv1d(chans[-1], latent_channels, kernel_size=1))

        self.encoder = nn.Sequential(*encoder)

        #--------------------------------------------- 
        with torch.no_grad():
            encoder_shape = self.encoder(torch.zeros(1, in_channels, input_length)).shape[1:]

        self.encoder_shape = tuple(encoder_shape)

        #---------------
        if latent_dim is not None : 
            latent_size = latent_dim 
        else :
            latent_size = encoder_shape.numel()

        if(latent_size >= in_channels * input_length):
            raise ValueError(f"no bottleneck : latent size {latent_size} >= input {in_channels} x {input_length} "
                             f"(increase n_layers, decrease latent_channels or set latent_dim)")

        #--------------------------------------------- DENSE BOTTLENECK
        if latent_dim is not None:

            flat = encoder_shape.numel()

            self.to_latent   = nn.Sequential(nn.Flatten(), nn.Linear(flat, latent_dim))
            self.from_latent = nn.Sequential(nn.Linear(latent_dim, flat), ACTIVATIONS[activation]())

        #--------------------------------------------- DECODER
        decoder = []

        if latent_dim is None:
            decoder += [nn.Conv1d(latent_channels, chans[-1], kernel_size=1), ACTIVATIONS[activation]()]

        #---------------
        for i in reversed(range(n_layers)):

            (k, c_in, c_out) = (k_sizes[i], chans[i + 1], chans[i])

            if decode_method == "resize":
                layer_i = ResizeConv1d(c_in, c_out, scale_factor=2, kernel_size=k)
            else:
                layer_i = nn.ConvTranspose1d(c_in, c_out, k, stride=2, padding=k // 2, output_padding=1)

            decoder += [layer_i, ACTIVATIONS[activation]() if i != 0 else nn.Identity()]

        self.decoder = nn.Sequential(*decoder)

    #================================================================================#
    def encode(self, x):

        """
        x : (batch, in_channels, L) -> latent code
        """

        #---------------------------------------------
        z = self.encoder(x)

        if self.latent_dim is not None:
            z = self.to_latent(z)                                          # (batch, latent_dim)

        return z

    #================================================================================#
    def decode(self, z):

        """
        latent code -> reconstruction, possibly longer than the input (cropped in forward)
        """

        #---------------------------------------------
        if self.latent_dim is not None:
            z = self.from_latent(z).view(-1, *self.enc_shape)              # (batch, C, L_enc)

        return self.decoder(z)

    #================================================================================#
    def forward(self, x):

        """
        process the input signal through the network and return its reconstruction
        """

        #---------------------------------------------
        recon = self.decode(self.encode(x))

        return recon[..., : x.shape[-1]]

    #================================================================================#
    def training_step(self, batch):

        x = batch[0]
        return nn_func.mse_loss(self(x), x)

    #================================================================================#
    def anomaly_score(self, x):

        mse = ((self(x) - x) ** 2).mean(dim=2).amax(dim=1)
        return mse