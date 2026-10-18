"""
A configuration describes what we want to launch : model, dataset, split, etc. 

It does not describes the model's hyperparameters, specifics to the model 
and an OUTPUT saved in results/<name>/best_params.json once they are tuned.

model.params       : hyperparameters FIXED for this experiment (not tuned, override DEFAULTS)
model.search_space : overrides of the model SEARCH_SPACE for this experiment (optional)
"""

#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import os
import copy
import json

import tools


#============================================================================================================================#
#--------------------------------------------------------- CONSTANT ---------------------------------------------------------#
#============================================================================================================================#
DEFAULT_RUN = {

    "name"          : None,                 # default : <model_name>_<dataset>_<objective>
    "seed"          : 0,
    "output_dir"    : "./results",
    "device"        : "auto",

    "data" : {
        "dataset"   : None,                 # required : key of data.DATASETS
        "dir_path"  : "./datasets/",
        "params"    : {},                   # loader specific (nb_sample, path_index, ...)
    },

    "split" : {
      "train_ratio"      : 0.8,
      "validation_ratio" : 0.2,
    },

    "preprocessing" : {
        "scaler" : "minmax",                # "minmax" | "standard" | "none"
    },

    "model" : {
        "name"         : None,              # required : key of models.MODELS
        "params"       : {},
        "search_space" : {},
    },

    "training" : {
        "epochs"      : 100,
        "patience"    : 10,
        "num_workers" : 0,
    },

    "tuning" : {
        "enabled"               : True,
        "objective"             : "pr_auc",             # "pr_auc" | "roc_auc" (semi-supervised) | "val_loss" (unsupervised)
        "n_trials"              : 30,
        "method"                : "random",             # "tpe" | random" | "grid"
        "pruning"               : False,                # optuna only & objective "val_loss" only : stops bad trials early
        "pruner_startup_trials" : 5, 
        "pruner_warmup_steps"   : 5
    },

    "threshold" : {
        "method" : "mean_std",              # "mean_std" | "quantile" | "youden"
        "value"  : 3.0,
    },

    "evaluation" : {
        "n_seeds" : 3,                      # final trainings with the tuned hyperparameters
    },
}

#============================================================================================================================#
#-------------------------------------------------------- FUNCTIONS ---------------------------------------------------------#
#============================================================================================================================#
def load_run(config_path: str) -> dict:

    """
    Load an experiment JSON file, complete it with DEFAULT_RUN and validate it.

    Args
    ----------
    config_path : path to the experiment JSON file

    Returns
    -------
    dict : complete experiment configuration
    """

    #---------------------------------------------
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"{config_path} : No such file or directory")

    with open(config_path, 'r', encoding='utf-8') as f:
        user_cfg = json.load(f)

    config = merge(DEFAULT_RUN, user_cfg, path="")
    config["config_path"] = config_path

    validate(config)

    #--------------------
    if config["name"] is None:
        dataset        = config["data"]["params"].get("subset", config["data"]["dataset"])
        config["name"] = f"{config['model']['name']}_{dataset}_{config['tuning']['objective']}"

    return config

#================================================================================#
def merge(default: dict, user: dict, path: str) -> dict:

    """
    Merge DEFAULT_RUN with the given configuration, based on a recursive merge method 
        - Unknown keys raise an error 

    Args
    ----------
    defaults : (dict)
    user     : (dict)
    path     : (str)

    Return
    ----------
    merged_config : (dict) complete configuration from merging `defaults` and `user`
    """

    #---------------------------------------------
    merged_config = copy.deepcopy(default)

    for (key, value) in user.items():

        if key == "description":
            merged_config[key] = value
            continue

        if key not in default:
            raise KeyError(f"unknown config key : '{path}{key}' (expected in {sorted(default)})")

        #--------------------
        if isinstance(default[key], dict) and default[key] and isinstance(value, dict):

            merged_config[key] = merge(default[key], value, path=f"{path}{key}.")

        else:
            merged_config[key] = value

    return merged_config

#================================================================================#
def validate(config : dict) -> None:

    """
    Validate a configuration : 
        
        - data.dataset is given
        - models.name is given
        - tuning.objective is correctly given
            - damage signals in validation if objective is not "val_loss"

        - evaluation.n_seeds is positive

    Args
    ----------
    config : (dict) configurtaion dictionnary to validate
    """
    
    #---------------------------------------------
    if not config["data"]["dataset"]:
        raise ValueError("data.dataset is required")

    #--------------------
    if not config["model"]["name"]:
        raise ValueError("model.name is required")

    #--------------------
    objective = config["tuning"]["objective"]

    if objective not in ("pr_auc", "roc_auc", "val_loss"):
        raise ValueError(f"tuning.objective must be 'pr_auc', 'roc_auc' or 'val_loss' : {objective}")

    if (objective != "val_loss") and (config["split"]["validation_ratio"] <= 0):
        
        raise ValueError(f"tuning.objective = '{objective}' needs damaged signals in the validation set : "
                         f"use objective 'val_loss'")

    #--------------------
    if config["evaluation"]["n_seeds"] < 1:
        raise ValueError("evaluation.n_seeds must be >= 1")

#================================================================================#
def print_config(config: dict) -> None:

    """
    print the experiment configuration

    Args
    ----------
    config : (dict) configurtaion dictionnary to print

    Returns
    ----------
    None

    Examples
    ----------
    #--------------- RUN : AE_OGW_val_loss ----------------#
        ->
        ->
        ->
        ->
    #------------------------------------------------------#
    """

    #---------------------------------------------
    tools.print_block(f"RUN : {config['name']}", config, exclude=("config_path",))

#================================================================================#
def print_HP(hp : dict, trial : int = -1) -> None:

    """
    print the hyperparameters in `hp`

    Args
    ----------
    hp    : (dict) dictionnary containing the hyperparametesr to print
    trial : (int) trial index for tuning display

    Returns
    ----------
    None

    Examples
    ----------
    #--------------- HYPERPARMETERS ----------------#
     -> batch : 32
     -> epoch : 100 
    #-----------------------------------------------#
    """

    #---------------------------------------------
    if trial < 0 :
        tools.print_block("HYPERPARAMETERS", hp)

    else :
        tools.print_block(f"Trial {trial} HP", hp)