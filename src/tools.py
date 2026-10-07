#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import os
import glob
import json
import random
import torch
import pickle
import h5py

import numpy as np

from datetime import datetime
from sklearn.base import TransformerMixin


#============================================================================================================================#
#--------------------------------------------------------- FUNCTION ---------------------------------------------------------#
#============================================================================================================================#
def set_seed(seed: int) -> None:

    """
    Seed python, numpy and torch for reproducible runs.

    Parameters
    ----------
    seed : random seed
    """

    #---------------------------------------------
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

#================================================================================#
def get_device(name: str = "auto") -> torch.device:

    """
    Return the torch device : "auto" -> cuda if available else cpu
    """

    #---------------------------------------------
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(name)

#================================================================================#
def save_json(obj, path: str) -> None:

    """
    Save a dict as an indented JSON file (numpy types converted)
    """

    #---------------------------------------------
    def convert(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, default=convert)

#================================================================================#
def load_json(path: str) -> dict:

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

#================================================================================#
def _parse_ts(path):
    try:
        return datetime.strptime(os.path.basename(path), "%Y%m%dT%H%M%S")
    except ValueError:
        return datetime.min
        
#================================================================================#
def sort_timestamp_data(state_dir : str, n_cycles) -> list:

    """
    Extract and sort a certain number of subdirectories from state_dir directory

    Parameters
    ----------
    state_dir : state directory form which we want to extract the data
    n_cycles  : number of subdirectories wanted to be extracted form state_dir

    Returns
    ----------
    subdirs : list of the subdirectories extracted
    """

    #---------------------------------------------
    subdirs = [d for d in glob.glob(os.path.join(state_dir, "*")) if os.path.isdir(d)]
    subdirs = sorted(subdirs, key=lambda p: (_parse_ts(p), os.path.basename(p)))
 
    if n_cycles is not None:
        subdirs = subdirs[:n_cycles]
 
    if not subdirs:
        print(f"[OGW] no timestamped folder found in : {state_dir}")
        return {}

    return subdirs

#================================================================================#
def memory_estimate(n_subdirs, n_keep_pitch, n_freqs) -> float:

    """
    Estimation of te needed memory that will be loaded

    Parameters
    ----------
    n_subdirs       : subdirectories to estimate the volume
    n_keep_pitch    : true if we want to keep the pitch else false
    n_freqs         : frequences to analyse

    Returns
    ----------
    gb : estimated gigabytes

    """

    #---------------------------------------------
    probe = sorted(glob.glob(os.path.join(n_subdirs[0], "*.h5")))

    if probe:

        with h5py.File(probe[0], "r") as f:
            p, n = f["pitchcatch/catch"].shape
        
        n_f = len(n_freqs) if n_freqs is not None else len(probe)
        gb  = len(n_subdirs) * n_f * p * n * 8 * (2 if n_keep_pitch else 1) / 1e9
        
        print(f"[OGW] approx memory for signals : {gb:.2f} GB "
                f"(use channel=<int> or freqs=[...] to shrink)")

        return gb

#================================================================================#
def safe_predict(model: torch.nn.Module,
                 tensor_cpu: torch.Tensor,
                 batch_size: int = 128) -> torch.Tensor:
    """
    Batched inference to avoid VRAM saturation.

    Parameters
    ----------
    model      : trained PyTorch model
    tensor_cpu : input tensor (kept on CPU, moved batch by batch)
    batch_size : number of samples per batch

    Returns
    ----------
    Reconstructions / predictions concatenated on CPU
    """

    #---------------------------------------------
    model.eval()
    device  = next(model.parameters()).device
    outputs = []

    with torch.no_grad():
        for i in range(0, len(tensor_cpu), batch_size):
            batch = tensor_cpu[i : i + batch_size].to(device)
            outputs.append(model(batch).cpu())

    return torch.cat(outputs, dim=0)

#================================================================================#
def save_model(model, scaler: TransformerMixin, model_name: str = "", models_dir: str = "./models/") -> None:

    """
    Save a model (.keras for Keras, .pth for PyTorch) along with its scaler.
    
    Parameters
    ----------
    model           : the model to save (Keras or PyTorch)
    scaler          : model's scaler tool
    model_name      : name of the model 
    models_dir      : output directory
    """

    #---------------------------------------------
    os.makedirs(models_dir, exist_ok=True)

    if model_name == "":

        sufix = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        base_name = f"model_{sufix}"

    else:
        base_name = f"{model_name}"

    #------------------------------
    print(f" -> Saving model ...", end="", flush=True)

    if type(model).__module__.startswith("keras"):
        
        model_path = os.path.join(models_dir, f"{base_name}.keras")
        model.save(model_path)

    elif isinstance(model, torch.nn.Module):
        
        model_path = os.path.join(models_dir, f"{base_name}.pth")
        torch.save(model.state_dict(), model_path)

    else:
        print("Error model type not recognized")

    print(f"Done ({model_path})")

    #------------------------------
    print(f" -> Saving model's scaler ...", end="", flush=True)
    
    scaler_path = os.path.join(models_dir, f"{base_name}_scaler.pkl")
    
    with open(scaler_path, 'wb') as file:
        pickle.dump(scaler, file)
    
    print(f"Done ({scaler_path})")

#================================================================================#
def load_model(model_path: str, 
               model_type: str, 
               model_class: type, 
               scaler_path: str, 
               model_kwargs: dict = None) -> tuple[object, TransformerMixin]:

    """
    Load a model (.keras for Keras, .pth for PyTorch) along with its scaler.

    Parameters
    ----------
    model_path : path to the saved model file.
    model_type : type of the model ("Keras" or "PyTorch")
    model_class : class of the PyTorch model (if model_type is "PyTorch")
    scaler_path : path to the saved scaler file.
    model_kwargs : dictionary of keyword arguments to initialize the PyTorch model (if model_type is "PyTorch")
    """

    #---------------------------------------------
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    #------------------------------
    if model_kwargs is None:
        model_kwargs = {}

    print(f"Loading model ...", end="", flush=True)

    if model_type == "Keras":
        from keras.models import load_model
        model = load_model(model_path)

    elif model_type == "PyTorch":
        if model_class is None:
            raise ValueError("model_class must be provided for PyTorch models.")
        
        model = model_class(**model_kwargs)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

    else:
        raise ValueError(f"Model type '{model_type}' not recognized.")

    print(f" Done ({model_path})")

    #------------------------------
    print(f"Loading scaler ...", end="", flush=True)
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    print(f" Done ({scaler_path})")

    return model, scaler
        

    