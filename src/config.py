#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import os
import json


CONFIG_PATH = "./config/config.json"

#============================================================================================================================#
#-------------------------------------------------------- FUNCTIONS ---------------------------------------------------------#
#============================================================================================================================#
def get_config(config_path: str = CONFIG_PATH) -> dict:

    """
    Load the configuration parameters from a JSON file.

    Parameters
    ----------
    config_path : path to the configuration JSON file.
    
    Returns
    -------
    dict : dictionary containing the configuration parameters.
    """

    #---------------------------------------------
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)

    return config_data

#================================================================================#
def set_config(config_data : dict) -> None :

    """
    set the global parameters with the data from the configuration file

    Parameters
    ----------
    config_data : dictionary containing the configuration parameters.
    """

    #---------------------------------------------
    global EPOCHS, BATCH_SIZE, LEARNING_RATE, LOSS_FUNCTION, OPTIMIZER
    global DATASET, DATAS_DIR
    global MODEL, MODELS_DIR, MODEL_NAME

    model           = config_data.get("model", {})
    data            = config_data.get("data", {})
    hyperparameters = config_data.get("hyperparameters", {})

    #------------------------------
    DATASET         = data.get("dataset", "default_dataset")
    DATAS_DIR       = data.get("dir_path", "./datasets/")

    MODEL           = model.get("type", "default_model")
    MODELS_DIR      = model.get("dir_path", "./models/")
    MODEL_NAME      = MODEL + "_" + DATASET + "_SHM"

    EPOCHS          = hyperparameters.get("epochs", 10)
    BATCH_SIZE      = hyperparameters.get("batch_size", 32)
    LEARNING_RATE   = hyperparameters.get("learning_rate", 0.001)
    LOSS_FUNCTION   = hyperparameters.get("loss", "mse")
    OPTIMIZER       = hyperparameters.get("optimizer", "adam")

#================================================================================#
def configuration() -> None :

    """
    print all configuration parameters from the configuration file
    """

    #---------------------------------------------
    print("\n#--------------- CONFIGURATION ----------------#")

    model()
    datas()
    hyperparameters()

    print("#----------------------------------------------#\n")

#================================================================================#
def model() -> None :

    """
    print model configuration from the configuration file
    """

    #---------------------------------------------
    print("\n#----------- MODEL ------------#")

    print(f"model          : {MODEL}")
    print(f"model name     : {MODEL_NAME}")

    print("#------------------------------#\n")

#================================================================================#
def datas() -> None :

    """
    print data configuration from the configuration file
    """

    #---------------------------------------------
    print("\n#----------- DATA ------------#")

    print(f"dataset        : {DATASET}")
    print(f"data directory : {DATAS_DIR}")

    print("#------------------------------#\n")

#================================================================================#
def hyperparameters() -> None :

    """
    print hyperparameters configuration from the configuration file
    """

    #---------------------------------------------
    print("\n#------- HYPERPARAMETERS --------#")

    print(f"Epochs          : {EPOCHS}")
    print(f"Batch size      : {BATCH_SIZE}")
    print(f"Learning rate   : {LEARNING_RATE}")
    print(f"Loss function   : {LOSS_FUNCTION}")
    print(f"Optimizer       : {OPTIMIZER}")

    print("#------------------------------#\n")

#============================================================================================================================#
#--------------------------------------------------------- CONSTANT ---------------------------------------------------------#
#============================================================================================================================#
DATAS_DIR   = "./datasets/"
MODELS_DIR  = "./models/"

#---------------------------------------------
EPOCHS          = 10
BATCH_SIZE      = 16
LEARNING_RATE   = 0.001

#---------------------------------------------
MODEL           = "CAE"
MODEL_NAME      = "CAE_model"