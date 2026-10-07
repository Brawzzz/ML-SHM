#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import tools
import setup as stp


#============================================================================================================================#
#---------------------------------------------------------- CLASS -----------------------------------------------------------#
#============================================================================================================================#
class ConvT1d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):

        super(ConvT1d, self).__init__()
        
        self.convT = nn.ConvTranspose1d(
            in_channels     = in_channels, 
            out_channels    = out_channels, 
            kernel_size     = kernel_size, 
            stride          = stride, 
            padding         = padding, 
            output_padding  = output_padding
        )
        self.activation = nn.ReLU()

    #------------------------------
    def forward(self, x):
        x = self.convT(x)
        x = self.activation(x)
        return x

#================================================================================#
class ResizeConv1d(nn.Module):
    
    def __init__(self, in_channels, out_channels, scale_factor=2, kernel_size=3):

        super(ResizeConv1d, self).__init__()
        
        self.upsample   = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.conv       = nn.Conv1d(
            in_channels     = in_channels, 
            out_channels    = out_channels, 
            kernel_size     = kernel_size, 
            padding         = kernel_size // 2 
        )
        
        self.activation = nn.ReLU()

    #------------------------------
    def forward(self, x):

        x = self.upsample(x)
        x = self.conv(x)
        x = self.activation(x)

        return x

#================================================================================#
class ConvAutoEncoder(nn.Module):

    """
    This class define the architecture for a Convolutional Auto-Encodeur CAE, based on torch.nn.Module

    Use 1D convolution to analyse temporal signal form SHM systems
    """

    #------------------------------
    def __init__(self, signal_length: int, decode_method: str = 'resize'):

        """
        Parameters
        ----------
        signal_length   : input signal length
        decode_method   : 'resize' for Resize-Conv / 'transpose' for ConvTranspose
        """

        super(ConvAutoEncoder, self).__init__()

        #---------------------------------------------
        self.encoder = nn.Sequential(

            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        #---------------------------------------------
        if decode_method == 'transpose':

            self.decoder = nn.Sequential(
                
                nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),

                nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.ReLU(),

                nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=7, stride=2, padding=3, output_padding=1)
            )
        
        #---------------------------------------------
        elif decode_method == 'resize':

            self.decoder = nn.Sequential(

                ResizeConv1d(in_channels=64, out_channels=32, scale_factor=2, kernel_size=3),
                ResizeConv1d(in_channels=32, out_channels=16, scale_factor=2, kernel_size=5),

                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(in_channels=16, out_channels=1, kernel_size=7, padding=3)
            )

        #---------------------------------------------
        else:
            raise ValueError(f"decoder_method must be 'transpose' or 'resize' : decoder_method : {decode_method}")

    #================================================================================#
    def forward(self, x):

        """
        process the input signal through the network

        Parameter
        ---------
        x : the input signal (3d)

        Returns
        ---------
        x : the reconstruction signal obtained by the network
        """

        #---------------------------------------------
        x = self.encoder(x)
        x = self.decoder(x)
        return x

#============================================================================================================================#
#--------------------------------------------------------- FUNCTION ---------------------------------------------------------#
#============================================================================================================================#
def CAE_train(X_uncrack: np.ndarray, X_crack: np.ndarray) -> tuple[nn.Module, float, np.ndarray, list, np.ndarray, np.ndarray]:

    """
    Convolutional Auto-Encoder training and evaluation function.

    Parameters
    ----------
    X_uncrack   : Input signals (NumPy array)
    X_crack     : Labels(NumPy array)
    
    Returns
    ----------
    model               : Trained autoencoder model.
    warning_threshold   : Alert threshold based on reconstruction error.
    crack_recon         : Reconstructed crack signals.
    train_losses        : Training losses.
    """

    #--------------------------------------------- 
    print(f" -> Healthy signals for training : {len(X_uncrack)}")
    print(f" -> Cracks signals for test      : {len(X_crack)}\n")

    (X_train, X_val) = train_test_split(X_uncrack, test_size=0.2, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" -> Training set on : {device}\n")
    
    #------------------------------
    tensor_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    tensor_val   = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    tensor_crack = torch.tensor(X_crack, dtype=torch.float32).unsqueeze(1)

    #------------------------------
    dataset     = TensorDataset(tensor_train, tensor_train)
    dataloader  = DataLoader(dataset, batch_size=stp.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    #------------------------------
    signal_size = X_uncrack.shape[1]
    model       = ConvAutoEncoder(signal_size, decode_method='resize').to(device)

    #------------------------------
    if stp.LOSS_FUNCTION == "mse":
        criterion   = nn.MSELoss()

    elif stp.LOSS_FUNCTION == "bce":
        criterion   = nn.BCELoss()

    else:
        criterion   = nn.MSELoss()

    #------------------------------
    if stp.OPTIMIZER == "adam":
        optimizer   = optim.Adam(model.parameters(), lr=stp.LEARNING_RATE)

    elif stp.OPTIMIZER == "sgd":
        optimizer   = optim.SGD(model.parameters(), lr=stp.LEARNING_RATE)

    else : 
        optimizer   = optim.Adam(model.parameters(), lr=stp.LEARNING_RATE)
    
    #===================================================================#
    #---------------------------- TRAINING -----------------------------#
    #===================================================================#
    pbar = tqdm(range(stp.EPOCHS), desc="Training CAE", unit="epoch")

    model.train()
    train_losses = []
    #------------------------------
    for epoch in range(stp.EPOCHS):

        train_loss = 0.0

        #---------------
        for batch_x, batch_y in dataloader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
        
            optimizer.zero_grad()        

            outputs = model(batch_x)
            loss    = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(dataloader.dataset)
        train_losses.append(train_loss)

        pbar.set_postfix({'Loss MSE': f"{train_loss:.5f}"})
        pbar.update(1)
        
    pbar.close()

    #===================================================================#
    #--------------------------- EVALUATION ----------------------------#
    #===================================================================# 
    healthy_recon = tools.safe_predict(model, tensor_val, batch_size=stp.BATCH_SIZE).squeeze(1).numpy()
    crack_recon   = tools.safe_predict(model, tensor_crack, batch_size=stp.BATCH_SIZE).squeeze(1).numpy()
    
    healthy_mse = np.mean(np.square(X_val - healthy_recon), axis=1)
    crack_mse   = np.mean(np.square(X_crack - crack_recon), axis=1)

    warning_threshold = np.mean(healthy_mse) + 3 * np.std(healthy_mse)

    #------------------------------
    print("\nRECONSTRUCTION MSE CAE :")
    print(f" -> Health MSE   : {np.mean(healthy_mse):.5f}")
    print(f" -> Crack MSE    : {np.mean(crack_mse):.5f}")
    print(f"\nWarning threshold set at  : {warning_threshold:.5f}")

    return(model, warning_threshold, crack_recon, train_losses, healthy_mse, crack_mse)

#================================================================================#
def CAE_inference(model: nn.Module, 
                  X_input: np.ndarray, 
                  n_threshold: float, 
                  n_batch_size: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Inference phase on single or multiple input signals

    Parameters
    ----------
    model      : Trained CAE
    X_input    : Input signals (NumPy array) => 1D / 2D (single / multiple signals).
    threshold  : Warning threshold form training phase
    batch_size : Size of batch in case of memory leak

    Returns
    ----------
    recons      : Reconstruted signals of each input signals
    mse_errors  : MSE error corresponding to the reconstructions errors of each input signals
    anomalies   : boolean tab where 1 = Anomalie and 0 = Healthy).
    """

    #---------------------------------------------    
    if X_input.ndim == 1:
        X_input = np.expand_dims(X_input, axis=0)
        
    tensor_input = torch.tensor(X_input, dtype=torch.float32).unsqueeze(1)

    #------------------------------
    recon_tensor = tools.safe_predict(model=model, tensor_cpu=tensor_input, batch_size=n_batch_size)
                
    recons      = recon_tensor.squeeze(1).numpy()
    mse_errors  = np.mean(np.square(X_input - recons), axis=1)
    anomalies   = (mse_errors > n_threshold).astype(int)
    
    return(recons, mse_errors, anomalies)

#================================================================================#
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