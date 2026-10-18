#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import os
import tqdm
import glob

import numpy as np 
import pandas as pd
import pickle as pkl

from .GWDataset import GWDataset


#============================================================================================================================#
#--------------------------------------------------------- CONSTANT ---------------------------------------------------------#
#============================================================================================================================#
UTAH_DIR             = "./datasets/UTAH"
UTAH_FILES_PREFIX    = "measurements_"
UTAH_FILES_EXTENSION = ".pickle"

DAMAGE_START = "2021_04"  
SETUP_START  = "2018_12"

#================================================================================#
def _utah_dir() -> str:
    return os.path.join(UTAH_DIR, "UTAH")

#================================================================================#
def file_month(path: str) -> str:

    """
    '.../measurements_2019_05.pickle' -> '2019_05'
    """
    #---------------------------------------------
    name = os.path.basename(path)
    return name[len(UTAH_FILES_PREFIX) : len(UTAH_FILES_PREFIX) + 7]

#============================================================================================================================#
#--------------------------------------------------------- FUNCTION ---------------------------------------------------------#
#============================================================================================================================#
def UTAH_load(nb_sample         : int  = 5,
              path_index        : int  = 3,
              dir_path          : str  = None,
              block_size        : int  = None,
              drop_undocumented : bool = True,
              start             : str  = SETUP_START,
              step              : int  = 1) -> GWDataset:

    """
    Load the UTAH - SHM dataset as a raw GWDataset.

    Each selected file is read, and the signals of ONE actuator-receiver path are kept. 
    Signals are returned unscaled : normalization must be fitted only on the train set, after the split.

    Each file (measurements_20xx_xx.pickle) is one month of acquisitions

    Args
    ----------
    nb_sample   : (int) number of files in which the data are extracted -- defaults : nb_sample = 5
    path_index  : (int) index of the analyse path (a pair of sensor) -- defaults : path_index = 3
    dir_path    : (str) path to the data directory -- defaults : dir_path = None
    block_size  : (int) number of consecutive measurements per group  -- defaults : block_size = None
                        if block_size = None -> one group per file (= 1 month)

    Return:
    ----------
    raw_dataset : GWDataset oject containing 

        X      : signals of the selected path -- shape (n_signals, signal_length)
        y      : signal's labels
                    - 0 = healthy (damage tag == 0)
                    - 1 = damaged (damage tag > 0)

        groups : acquisition block of each signal, chronological ids (0 = oldest)
                 ids restart at 0 in each file, then are shifted to be unique across files

        meta   : signal's metadata, one row per signal -- columns : file, damage, weather, temperature

    Notes
    ----------
    splits() needs at least 2 groups per class (healthy / damaged).
    If damaged signals appear in only a few files, set block_size to get more groups.

    """

    #---------------------------------------------
    if dir_path is None:
        dir_path = UTAH_DIR

    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"[UTAH] directory not found : {dir_path}")
    
    files = UTAH_files(nb_sample = nb_sample, 
                       data_dir  = dir_path, 
                       start     = start)

    if not files:
        raise FileNotFoundError(f"no UTAH file found in : {dir_path}")

    #------------------------------
    paths                 = np.atleast_1d(path_index)
    next_group_id         = 0
    (X, y, groups, metas) = ([], [], [], [])

    for file in tqdm.tqdm(files, desc="Loading UTAH dataset", unit="file"):

        with open(file, 'rb') as f:
            dataset = pkl.load(f)

        damages = np.asarray(dataset['damage tag'])

        #--------------- signals to keep
        keep = (np.arange(len(damages)) % step == 0)                  # sub-sampling

        if drop_undocumented and file_month(file) < DAMAGE_START:
            keep &= (damages == 0)                                     # undocumented tags before D1

        idx = np.flatnonzero(keep)
        if len(idx) == 0:
            continue

        #--------------- groups
        if block_size:
            local_group_id = np.arange(len(idx)) // block_size
        else:
            local_group_id = np.zeros(len(idx), dtype=int)

        X.append(dataset['guided wave'][idx][:, paths, :])
        y.append((damages[idx] > 0).astype(int))

        groups.append(next_group_id + local_group_id)
        next_group_id += local_group_id.max() + 1

        metas.append(pd.DataFrame({"file"        : os.path.basename(file),
                                   "damage"      : damages[idx],
                                   "weather"     : np.asarray(dataset['weather tag'])[idx],
                                   "temperature" : np.asarray(dataset['temperature'])[idx]}))

    #------------------------------
    raw_dataset = GWDataset(name   = "UTAH",
                            X      = np.vstack(X),
                            y      = np.concatenate(y),
                            groups = np.concatenate(groups),
                            meta   = pd.concat(metas, ignore_index=True))

    return raw_dataset

#================================================================================#
def UTAH_files(nb_sample : int = 5, data_dir : str = None, start : str = None) -> list:

    """
    Return nb_sample paths within data_dir directory.
    The selected files are automatically regularly spaced in time to ensure temporal diversity.

    Args
    ----------
    nb_sample : number of files to extract from data_dir -- default : nb_sample = 5
    data_dir  : directory where the UTAH data files are stored -- default : data_dir = UTAH_DIR = "./datasets/UTAH/"

    Returns
    ----------
    files : list containing nb_sample paths in chronological order
    """

    #---------------------------------------------
    if data_dir is None:
        data_dir = _utah_dir()

    all_files = sorted(glob.glob(os.path.join(data_dir, f"{UTAH_FILES_PREFIX}*{UTAH_FILES_EXTENSION}")))

    if start is not None:
        all_files = [f for f in all_files if file_month(f) >= start]

    if not all_files:
        print(f"[UTAH] no file found in : {data_dir} --- check file format : measurements_20xx_xx.pickle")
        return []

    #------------------------------
    if nb_sample >= len(all_files) or nb_sample <= 0:
        return all_files

    idx = np.linspace(0, len(all_files) - 1, nb_sample).round().astype(int)
    idx = sorted(set(idx))

    return [all_files[i] for i in idx]

#================================================================================#
def UTAH_rename_files(data_dir : str = None) -> None:
    
    """
    rename the UTAH files in the data_dir directory to ensure a consistent naming convention.

    Args
    ----------
    data_dir : directory where the UTAH data files are stored (default : UTAH_DIR = "<DATAS_DIR>/UTAH")
    """

    #---------------------------------------------
    if data_dir is None:
        data_dir = _utah_dir()

    all_files = sorted(glob.glob(os.path.join(data_dir, f"measurements *{UTAH_FILES_EXTENSION}")))

    if not all_files:
        print(f"[UTAH] no file found in : {data_dir}")
    
    for file in all_files:

        print(f"file            : {file}")
        os.rename(file, file.replace("measurements ", "measurements_")) 
        print(f"renamed file    : {file}")