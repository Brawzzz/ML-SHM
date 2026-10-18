#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import os
import glob
import torch
import h5py
import json
import random

import numpy as np

from datetime import datetime


#============================================================================================================================#
#--------------------------------------------------------- CONSTANT ---------------------------------------------------------#
#============================================================================================================================#
LINE_WIDTH = 60

#============================================================================================================================#
#--------------------------------------------------------- FUNCTION ---------------------------------------------------------#
#============================================================================================================================#
def set_seed(seed: int) -> None:

    """
    Seed python, numpy and torch for reproducible runs.

    Args
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
    Return the wanted torch device 

    Args
    ----------
    name : (str) name of the wanted device : defaults : "auto" -> if cuda avaible eles cpu

    Return
    ----------
    torch.device
    """

    #---------------------------------------------
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(name)

#================================================================================#
def _parse_ts(path):
    try:
        return datetime.strptime(os.path.basename(path), "%Y%m%dT%H%M%S")
    except ValueError:
        return datetime.min

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
def sort_timestamp_data(state_dir : str, n_cycles) -> list:

    """
    Extract and sort a certain number of subdirectories from state_dir directory

    Args
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
def print_dict(_dict  : dict, 
               level   : int = 0, 
               exclude : tuple = ()) -> None:

    """
    Print a dictionary as an indented tree, nested dictionaries unfolded.
    The ':' are aligned within each level.

    Parameters
    ----------
    _dict   : (dict) dictionary to print
    level   : (int) indentation level -- 4 spaces per level -- defaults level = 0
    exclude : (tuple) keys not printed (only at this level) -- defaults exclude = ()

    Example
    ----------
     -> name : CAE_UTAH_lat32
     -> data
         -> dataset  : UTAH
         -> dir_path : D:/STG3/datasets/
    """

    #---------------------------------------------
    keys = [k for k in _dict if k not in exclude]

    if not keys:
        return

    leaves = [k for k in keys if not (isinstance(_dict[k], dict) and _dict[k])]
    width  = max((len(str(k)) for k in leaves), default=0)
    pad    = " " + "    " * level

    #---------------------------------------------
    for key in keys:

        value = _dict[key]

        if isinstance(value, dict) and value:
            print(f"{pad} -> {key}")
            print_dict(value, level + 1)

        else:
            print(f"{pad} -> {str(key):<{width}} : {value}")

#================================================================================#
def print_block(title   : str, 
                _dict   : dict, 
                exclude : tuple = ()) -> None:

    """
    Print `dict` between two banners of LINE_WIDTH characters, the title centered in the first one.

    Parameters
    ----------
    title   : (str) title of the block
    _dict   : (dict) dictionary to print
    exclude : (tuple) top-level keys not printed -- defaults exclude = ()

    Example
    ----------
    #-------------------- HYPERPARAMETERS ---------------------#
     -> lr         : 0.001
     -> batch_size : 32
    #----------------------------------------------------------#
    """

    #---------------------------------------------
    print("\n#" + f" {title} ".center(LINE_WIDTH - 2, "-") + "#")
    print_dict(_dict, exclude=exclude)
    print("#" + "-" * (LINE_WIDTH - 2) + "#\n")