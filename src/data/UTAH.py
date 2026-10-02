#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import os
import tqdm

import pickle as pkl
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import tools
import setup as stp


#============================================================================================================================#
#--------------------------------------------------------- CONSTANT ---------------------------------------------------------#
#============================================================================================================================#
UTAH_FILES              = []
UTAH_DIR                = stp.DATAS_DIR + "UTAH/"
UTAH_FILES_PREFIX       = "measurements_"
UTAH_FILES_EXTENSION    = ".pickle"
UTAH_FILES_YEAR         = ["2018_", "2022_"]

#============================================================================================================================#
#--------------------------------------------------------- FUNCTION ---------------------------------------------------------#
#============================================================================================================================#
def UTAH_load(nb_sample : int, path_index : int = 3):

    """
    Extraction and prepration of the datas from UTHA database
    
    Parameters
    ----------
    data_file   : list of paths for the data file (file format : measurements_20xx_xx.pickle)
    path_index  : index of sensor to analyse (exemple : 3 --> path 5-4)

    Returns
    ----------
    X_train     : healthy dataset 
    X_test      : cracks dataset, 
    scaler      : scaler tools used for training phase  
    cracks_info : damage caracteristics
    """

    #------------------------------
    files       = []
    START_IDX   = 6

    for year in UTAH_FILES_YEAR:
        for i in range(START_IDX, START_IDX+nb_sample):

            sample_index    = str(i) if i > 9 else "0" + str(i)
            sample_name     = UTAH_FILES_PREFIX + year + sample_index + UTAH_FILES_EXTENSION
            sample_path     = os.path.join(UTAH_DIR, sample_name) 

            files.append(sample_path)

    #------------------------------
    valid_files = [file for file in files if os.path.exists(file)]
    
    if not valid_files:
        print("None valid files")
        return(None, None, None, None)

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