#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import os
import tqdm
import glob

import pickle as pkl
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import tools
import setup as stp


#============================================================================================================================#
#--------------------------------------------------------- CONSTANT ---------------------------------------------------------#
#============================================================================================================================#
UTAH_FILES_PREFIX       = "measurements_"
UTAH_FILES_EXTENSION    = ".pickle"

#================================================================================#
def _utah_dir() -> str:
    return os.path.join(stp.DATAS_DIR, "UTAH")

#============================================================================================================================#
#--------------------------------------------------------- FUNCTION ---------------------------------------------------------#
#============================================================================================================================#
def UTAH_load(nb_sample : int, path_index : int = 3):

    """
    Extraction and prepration of the datas from UTHA database
    
    Parameters
    ----------
    nb_sample   : number of sample to load
    path_index  : index of sensor to analyse (exemple : 3 --> path 5-4)

    Returns
    ----------
    X_train     : healthy dataset 
    X_test      : cracks dataset, 
    scaler      : scaler tools used for training phase  
    cracks_info : damage caracteristics
    """

    #------------------------------
    files       = UTAH_files(nb_sample=nb_sample, data_dir=_utah_dir())
    valid_files = [file for file in files if os.path.exists(file)]

    if not valid_files:
        print("None valid files")
        return(None, None, None, None)

    print("\n#----------------- UTAH FILES -----------------#\n")
    for file in valid_files : 
        print(f"file : {file}")
    print("")
    
    #------------------------------
    healthy_signals   = []
    cracks_signals    = []
    cracks_info       = []

    pbar = tqdm.tqdm(range(len(valid_files)), desc="Loading UTAH datas", unit="files")

    for file in valid_files:

        if not os.path.exists(file):
            print(f"No such file or directory : {file}")
            continue
            
        with open(file, 'rb') as f:
            dataset = pkl.load(f)

        #---------------
        signals = dataset['guided wave']
        damages = dataset['damage tag']
        weather = dataset['weather tag']
        temps   = dataset['temperature']

        #---------------
        idx_healty = np.where(damages == 0)[0]

        for i in idx_healty:
            healthy_signals.append(signals[i, path_index, :])

        #---------------
        idx_fissures = np.where(damages > 0)[0]

        for i in idx_fissures:

            cracks_signals.append(signals[i, path_index, :])

            context = f"Damge D{damages[i]} | Weather : {weather[i]} | Temp: {temps[i]:.1f}°C"
            cracks_info.append(context)

        #---------------
        pbar.update(1)

    pbar.close()

    #---------------------------------------------
    print(f"\nhealthy / cracks signals segmentation ...", end="", flush=True)

    X_train_raw = np.array(healthy_signals)
    X_test_raw  = np.array(cracks_signals)

    print(f"Done\n")

    print(f" -> healthy signals extracted \t: {len(X_train_raw)}")
    print(f" -> crack signals extracted \t: {len(X_test_raw)}\n")

    #---------------
    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train_raw)

    if len(X_test_raw) > 0:
        X_test = scaler.transform(X_test_raw)
    else:
        X_test = np.empty((0, X_train_raw.shape[1]))
        
    return(X_train, X_test, scaler, cracks_info)

#================================================================================#
def UTAH_files(nb_sample: int = 5, data_dir : str = None) -> list:

    """
    Return nb_sample paths within data_dir directory.
    The selected files are automatically regularly spaced in time to ensure temporal diversity.

    Parameters
    ----------
    nb_sample : number of files to extract from data_dir (default : 5)
    data_dir  : directory where the UTAH data files are stored (default : UTAH_DIR = "./datasets/UTAH/")

    Returns
    ----------
    files : list containing nb_sample paths in chronological order
    """

    #---------------------------------------------
    if data_dir is None:
        data_dir = _utah_dir()

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
def UTAH_rename_files(data_dir : str = None) -> None:
    
    """
    rename the UTAH files in the data_dir directory to ensure a consistent naming convention.

    Parameters
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