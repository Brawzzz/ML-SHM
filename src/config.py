#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import os
import copy
import json


#============================================================================================================================#
#--------------------------------------------------------- CONSTANT ---------------------------------------------------------#
#============================================================================================================================#

"""
An experiment config describes WHAT is run (model, dataset, split, budget), not the
model's hyperparameters : those belong to the model (DEFAULTS / SEARCH_SPACE in its
class) and their tuned values are an OUTPUT saved in results/<name>/best_params.json.

model.params       : hyperparameters FIXED for this experiment (not tuned, override DEFAULTS)
model.search_space : overrides of the model SEARCH_SPACE for this experiment (optional)
"""

DEFAULT_EXPERIMENT = {

    "name"          : None,                 # default : <model>_<dataset>_<objective>
    "seed"          : 0,
    "output_dir"    : "./results",
    "device"        : "auto",

    "data" : {
        "dataset"   : None,                 # required : key of data.DATASETS
        "dir_path"  : "./datasets/",
        "params"    : {},                   # loader specific (nb_sample, path_index, ...)
    },

    "split" : {
        "strategy"         : "group",       # "group" | "chrono" | "random"
        "val_size"         : 0.15,          # fraction of healthy signals
        "test_size"        : 0.15,          # fraction of healthy signals
        "damaged_val_size" : 0.3,           # fraction of damaged signals in val (rest in test)
    },

    "preprocessing" : {
        "scaler" : "minmax",                # "minmax" | "standard" | "none" (fitted on train only)
    },

    "model" : {
        "type"         : None,              # required : key of models.MODELS
        "params"       : {},
        "search_space" : {},
    },

    "training" : {
        "epochs"      : 100,                # maximum, early stopping on the healthy validation loss
        "patience"    : 10,
        "num_workers" : 0,
    },

    "tuning" : {
        "enabled"   : True,
        "objective" : "pr_auc",             # "pr_auc" | "roc_auc" (semi-supervised) | "val_loss" (unsupervised)
        "n_trials"  : 30,
        "timeout"   : None,                 # seconds
        "resume"    : False,                # keep trials in results/<name>/optuna.db and resume them
    },

    "threshold" : {
        "method" : "mean_std",              # "mean_std" (mean + value * std) | "quantile"
        "value"  : 3.0,
    },

    "evaluation" : {
        "n_seeds" : 3,                      # final trainings with the selected hyperparameters
    },
}

#============================================================================================================================#
#-------------------------------------------------------- FUNCTIONS ---------------------------------------------------------#
#============================================================================================================================#
def load_experiment(config_path: str) -> dict:

    """
    Load an experiment JSON file, complete it with DEFAULT_EXPERIMENT and validate it.

    Parameters
    ----------
    config_path : path to the experiment JSON file

    Returns
    -------
    dict : complete experiment configuration
    """

    #---------------------------------------------
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        user_cfg = json.load(f)

    cfg = _merge(DEFAULT_EXPERIMENT, user_cfg, path="")
    cfg["config_path"] = config_path

    _validate(cfg)

    if cfg["name"] is None:
        cfg["name"] = f"{cfg['model']['type']}_{cfg['data']['dataset']}_{cfg['tuning']['objective']}"

    return cfg

#================================================================================#
def _merge(default: dict, user: dict, path: str) -> dict:

    """
    Recursive merge, unknown keys raise an error (typo protection)
    """

    #---------------------------------------------
    out = copy.deepcopy(default)

    for (k, v) in user.items():

        if k == "description":
            out[k] = v
            continue

        if k not in default:
            raise KeyError(f"unknown config key : '{path}{k}' (expected one of {sorted(default)})")

        # free-form dicts (loader / model params) are taken as is
        if isinstance(default[k], dict) and default[k] and isinstance(v, dict):
            out[k] = _merge(default[k], v, path=f"{path}{k}.")
        else:
            out[k] = v

    return out

#================================================================================#
def _validate(cfg: dict) -> None:

    #---------------------------------------------
    if not cfg["data"]["dataset"]:
        raise ValueError("data.dataset is required")

    if not cfg["model"]["type"]:
        raise ValueError("model.type is required")

    objective = cfg["tuning"]["objective"]

    if objective not in ("pr_auc", "roc_auc", "val_loss"):
        raise ValueError(f"tuning.objective must be 'pr_auc', 'roc_auc' or 'val_loss' : {objective}")

    if objective != "val_loss" and cfg["split"]["damaged_val_size"] <= 0:
        raise ValueError(f"tuning.objective = '{objective}' needs damaged signals in the validation set : "
                         f"set split.damaged_val_size > 0 (or use objective 'val_loss')")

    if cfg["evaluation"]["n_seeds"] < 1:
        raise ValueError("evaluation.n_seeds must be >= 1")

#================================================================================#
def print_config(cfg: dict) -> None:

    """
    print the experiment configuration
    """

    #---------------------------------------------
    print(f"\n#--------------- EXPERIMENT : {cfg['name']} ----------------#")
    print(json.dumps({k: v for k, v in cfg.items() if k != "config_path"}, indent=4))
    print("#----------------------------------------------#\n")
