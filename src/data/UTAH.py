#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import os
import tqdm
import glob

import pickle as pkl
import numpy as np
import pandas as pd

from .base import SHMDataset


#============================================================================================================================#
#--------------------------------------------------------- CONSTANT ---------------------------------------------------------#
#============================================================================================================================#
UTAH_FILES_PREFIX       = "measurements_"
UTAH_FILES_EXTENSION    = ".pickle"

#================================================================================#
def _utah_dir(dir_path: str) -> str:
    return os.path.join(dir_path, "UTAH")

#============================================================================================================================#
#--------------------------------------------------------- FUNCTION ---------------------------------------------------------#
#============================================================================================================================#
def UTAH_load(dir_path: str, nb_sample: int = 20, path_index: int = 3, verbose: bool = True) -> SHMDataset:

    """
    Extraction of the RAW datas from the UTAH database (no normalisation, no split).

    Each file (one month of measurements) is one acquisition block : all its signals
    share the same `groups` id, so a block is never shared between train / val / test.

    Parameters
    ----------
    dir_path    : root datasets directory (the files are read in <dir_path>/UTAH)
    nb_sample   : number of files to load, regularly spaced in time (<= 0 -> all)
    path_index  : index of sensor path to analyse (exemple : 3 --> path 5-4)
    verbose     : print loading information

    Returns
    ----------
    SHMDataset : X (all signals), y (0 healthy / 1 damaged), groups (file index), meta
    """

    #------------------------------
    files       = UTAH_files(nb_sample=nb_sample, data_dir=_utah_dir(dir_path))
    valid_files = [file for file in files if os.path.exists(file)]

    if not valid_files:
        raise FileNotFoundError(f"[UTAH] no valid file found in : {_utah_dir(dir_path)}")

    if verbose:
        print("\n#----------------- UTAH FILES -----------------#\n")
        for file in valid_files:
            print(f"file : {file}")
        print("")

    #------------------------------
    (signals_list, labels, groups, meta_rows) = ([], [], [], [])

    for block, file in enumerate(tqdm.tqdm(valid_files, desc="Loading UTAH datas", unit="files", disable=not verbose)):

        with open(file, 'rb') as f:
            dataset = pkl.load(f)

        #---------------
        signals = np.asarray(dataset['guided wave'])
        damages = np.asarray(dataset['damage tag'])
        weather = np.asarray(dataset['weather tag'])
        temps   = np.asarray(dataset['temperature'])

        #---------------
        signals_list.append(signals[:, path_index, :])
        labels.append((damages > 0).astype(int))
        groups.append(np.full(len(damages), block))

        for i in range(len(damages)):
            meta_rows.append({
                "file"        : os.path.basename(file),
                "damage_tag"  : int(damages[i]),
                "weather"     : weather[i],
                "temperature" : float(temps[i]),
            })

    #---------------------------------------------
    ds = SHMDataset(name   = "UTAH",
                    X      = np.concatenate(signals_list, axis=0),
                    y      = np.concatenate(labels),
                    groups = np.concatenate(groups),
                    meta   = pd.DataFrame(meta_rows))

    if verbose:
        print(f"\n{ds.describe()}\n")

    return ds

#================================================================================#
def UTAH_files(nb_sample: int = 5, data_dir : str = "./datasets/UTAH") -> list:

    """
    Return nb_sample paths within data_dir directory.
    The selected files are automatically regularly spaced in time to ensure temporal diversity.

    Parameters
    ----------
    nb_sample : number of files to extract from data_dir (default : 5, <= 0 -> all)
    data_dir  : directory where the UTAH data files are stored

    Returns
    ----------
    files : list containing nb_sample paths in chronological order
    """

    #---------------------------------------------
    all_files = sorted(glob.glob(os.path.join(data_dir, f"{UTAH_FILES_PREFIX}*{UTAH_FILES_EXTENSION}")))

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
def UTAH_rename_files(data_dir : str = "./datasets/UTAH") -> None:

    """
    rename the UTAH files in the data_dir directory to ensure a consistent naming convention.

    Parameters
    ----------
    data_dir : directory where the UTAH data files are stored
    """

    #---------------------------------------------
    all_files = sorted(glob.glob(os.path.join(data_dir, f"measurements *{UTAH_FILES_EXTENSION}")))

    if not all_files:
        print(f"[UTAH] no file found in : {data_dir}")

    for file in all_files:

        print(f"file            : {file}")
        os.rename(file, file.replace("measurements ", "measurements_"))
        print(f"renamed file    : {file}")
