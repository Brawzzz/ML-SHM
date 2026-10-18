#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import os
import optuna
import functools

import pandas as pd

import tools

from models import BaseDetector
from Metrics import Metrics

from config import print_HP
from training import train_model, predict_scores


#============================================================================================================================#
#--------------------------------------------------------- CONSTANT ---------------------------------------------------------#
#============================================================================================================================#
OBJECTIVES = {
    "val_loss" : "minimize",    # unsupervised    : model loss on the HEALTHY validation signals
    "pr_auc"   : "maximize",    # semi-supervised : average precision on the validation set (needs damaged signals)
    "roc_auc"  : "maximize",    # semi-supervised : ROC-AUC on the validation set (needs damaged signals)
}

METHODS = ("tpe", "random", "grid")

#============================================================================================================================#
#--------------------------------------------------------- FUNCTION ---------------------------------------------------------#
#============================================================================================================================#
def tune(model_cls : type[BaseDetector], 
         data      : dict, 
         config    : dict, 
         out_dir   : str, 
         device    ) -> dict:

    """
    Loop for hyperparameter search
    
    Try different combinations, keep the one with the best validation score : 
        1 - training on train set
        2 - score on validation set
        3 - keep the config with the best score 

    Parameters
    ----------
    model_cls : (type[BaseDetector]) model class to tune (e.g. ConvAutoEncoder)
    data      : (dict) perpared data dictionnary from prepare() -- {"train" : (X, y), "validation" : (X, y), "test": (X, y) } 
    config    : (dict) experiment configuration
    out_dir   : (str) path to output directory : where to save the best params  
    device    : (torch.device) cpu or cuda

    Returns
    ----------
    best_params : (dict) complete hyperparameters of the best trial on the validation set :

                - tuned    : values drawn from the search space
                - fixed    : values imposed by the experiment config (model.params)
                - defaults : model DEFAULTS for every other hyperparameter
    """

    #---------------------------------------------
    tune_config = config["tuning"]
    objective   = tune_config["objective"]
    method      = tune_config.get("method", "random")

    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective : {objective} (expected one of {list(OBJECTIVES)})")
    
    if method not in METHODS:
        raise ValueError(f"unknown search method : {method} (expected one of {METHODS})")
    
    #--------------------
    all_hp_search = {**model_cls.SEARCH_SPACE, **config["model"].get("search_space", {})}

    hp_fixed    = config["model"].get("params", {})
    hp_search   = {name : values for name, values in all_hp_search.items() if name not in hp_fixed}
    
    #---------------------------------------------
    (best_value, best_params, trials) = search(model_cls = model_cls, 
                                               hp_search = hp_search, 
                                               hp_fixed  = hp_fixed,
                                               data      = data, 
                                               config    = config, 
                                               method    = method, 
                                               device    = device)

    trials.to_csv(os.path.join(out_dir, "trials.csv"), index=False)

    save_path = os.path.join(out_dir, "best_params.json")
    tune_dict = {"objective"  : objective, 
                 "best_value" : best_value, 
                 "n_trials"   : len(trials),
                 "params"     : best_params}
    
    tools.save_json(tune_dict, save_path)

    return best_params

#================================================================================#
def search(model_cls, 
           hp_search : dict, 
           hp_fixed  : dict,
           data      : dict, 
           config    : dict, 
           method    : str, 
           device) -> tuple:
 
    """
    Run the Optuna study with the sampler of `method`.
 
    Pruning (tuning.pruning = true --> objective = "val_loss") : the validation loss of each epoch is reported
    to Optuna, which stops a trial early when it is worse than the median of the previous trials at the same epoch.
    
    It is not used for "pr_auc" / "roc_auc", since the loss reported during training is not the objective.
    
    Args
    ----------
    model_cls : model class to tune (e.g. ConvAutoEncoder)
    hp_search : (dict) hyperparameters dictionnary to tune
    hp_fixed  : (dict) fixed hyperparameters dictionnary
    data      : (dict) prepared data
    config    : (dict) experiment configuration
    method    : (str) sampler type for optuna -- `tpe` | `grid` | `random`
    device    : (torch.device) cpu or cuda

    Returns
    ----------
    (best_value, best_params, trials) : (tuple) 
                                        - best objective value, 
                                        - complete best hyperparameters,
                                        - DataFrame with one row per trial (Optuna format)
    """
 
    #---------------------------------------------
    optuna.logging.set_verbosity(optuna.logging.WARNING)
 
    tune_config = config["tuning"]
    objective   = tune_config["objective"]
    seed        = config["seed"]
 
    use_pruning = (tune_config.get("pruning", False)) and (objective == "val_loss")
 
    if (tune_config.get("pruning", False)) and (not use_pruning):
        print(f" -> pruning ignored : only available with objective 'val_loss' (here '{objective}')")
 
    #---------------------------------------------
    study = optuna.create_study(study_name = config["name"],
                                direction  = OBJECTIVES[objective],
                                sampler    = make_sampler(method, hp_search, seed),
                                pruner     = make_pruner(pruning = use_pruning, tune_config = tune_config))
 
    study.set_user_attr("objective", objective)  
 
    #---------------------------------------------
    objective_fn = functools.partial(objective_function,
                                     model_cls   = model_cls,
                                     hp_search   = hp_search,
                                     hp_fixed    = hp_fixed,
                                     data        = data,
                                     config      = config,
                                     device      = device,
                                     use_pruning = use_pruning)
 
    # grid : the GridSampler stops the study by itself once every combination has been tried
    n_trials = None if method == "grid" else tune_config["n_trials"]
 
    study.optimize(objective_fn, n_trials=n_trials, callbacks=[log_trial])
 
    #---------------------------------------------
    completed = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    if not completed:
        raise RuntimeError("no Optuna trial completed (all pruned or failed)")
 
    best_params = model_cls.resolve_params({**study.best_params, **hp_fixed})
 
    return (study.best_value, best_params, trials_DF(study))

#================================================================================#
def objective_function(trial,
                       model_cls,
                       hp_search   : dict,
                       hp_fixed    : dict,
                       data        : dict,
                       config      : dict,
                       device,
                       use_pruning : bool) -> float:
 
    """
    Objective function of the Optuna study : ONE trial.
 
        1 - draw the hyperparameters of the search space
        2 - complete them with the fixed ones and the model DEFAULTS
        3 - train the model and return the objective on the validation set
 
    Args
    ----------
    trial       : (optuna.Trial) current trial, given by Optuna
    model_cls   : model class to tune
    hp_search   : (dict) hyperparameters dictionnary to tune
    hp_fixed    : (dict) fixed hyperparameters dictionnary
    data        : (dict) prepared data
    config      : (dict) experiment configuration
    device      : (torch.device) cpu or cuda
    use_pruning : (bool) report the validation loss of each epoch to Optuna
 
    Returns
    ----------
    value : (float) objective value on the validation set
    """
 
    #---------------------------------------------
    hp_sampled = {name : suggest(trial, name, space) for name, space in hp_search.items()}
    params     = model_cls.resolve_params({**hp_sampled, **hp_fixed})
 
    on_epoch_end = functools.partial(report_epoch, trial) if use_pruning else None
 
    #---------------------------------------------
    (value, history) = run_trial(model_cls    = model_cls,
                                 params       = params,
                                 data         = data,
                                 config       = config,
                                 seed         = config["seed"] + trial.number,
                                 device       = device,
                                 name         = f"{config['name']}_trial{trial.number}",
                                 on_epoch_end = on_epoch_end)
 
    trial.set_user_attr("best_epoch", history["best_epoch"])
 
    return value
 
#================================================================================#
def report_epoch(trial, epoch : int, model, val_loss : float) -> None:
 
    """
    Training callback for pruning : report the validation loss of the epoch to Optuna
    and stop the trial if the pruner decides it is not promising.
 
    Used through functools.partial(report_epoch, trial) -> callback(epoch, model, val_loss)

    Args
    ----------
    trail    : 
    epoch    : 
    val_loss : (float)

    Returns
    ----------
    None 
    """
 
    #---------------------------------------------
    trial.report(val_loss, step=epoch)
 
    if trial.should_prune():
        raise optuna.TrialPruned(f"pruned at epoch {epoch}")
 
#================================================================================#
def log_trial(study : optuna.Study, trial : optuna.trial.FrozenTrial) -> None:
 
    """
    Optuna callback, called after EACH trial (completed | pruned | failed)
    Print its result and the best value so far.

    Args
    ----------
    study : (optuna.Study)
    trial : (optuna.trial.FrozenTrial)

    Returns
    ----------
    None
    """
 
    #---------------------------------------------
    objective = study.user_attrs.get("objective", "value")
 
    try:
        best = f"{study.best_value:.4g}"
    except ValueError:
        best = "-"
 
    #--------------------
    if trial.state == optuna.trial.TrialState.PRUNED:
        print(f"\n--> trial {trial.number:>3} : pruned at epoch {trial.last_step} | best = {best}\n")
 
    elif trial.state == optuna.trial.TrialState.FAIL:
        print(f"\n--> trial {trial.number:>3} : failed | best = {best}\n")
 
    else:
        print(f"\n--> trial {trial.number:>3} : {objective} = {trial.value:.4g} | best = {best}\n")

#================================================================================#
def make_sampler(method: str, hp_search: dict, seed: int) -> optuna.samplers.BaseSampler:

    """
    Return a choosen optuna smapler depending on `method`

    Args
    ----------
    method    : (str)
    hp_search : (dict)
    seed      : (int)

    Returns
    ----------
    sampler : (optuna.samplers.BaseSampler)
    """

    #---------------------------------------------
    if method == "tpe":
        return optuna.samplers.TPESampler(seed=seed)

    if method == "random":
        return optuna.samplers.RandomSampler(seed=seed)

    if method == "grid":

        ranges = [n for n, s in hp_search.items() if isinstance(s, dict)]

        if ranges:
            raise ValueError(f"grid search needs lists of values, not ranges : {ranges}")
        return optuna.samplers.GridSampler({n: list(s) for n, s in hp_search.items()}, seed=seed)

    #--------------------
    raise ValueError(f"unknown search method : {method}")

#================================================================================#
def make_pruner(pruning : bool, tune_config : dict) -> optuna.pruners.BasePruner:

    """
    MedianPruner if pruning is used, NopPruner (never prunes) otherwise.
 
    Optional config keys (tuning.*) :
        n_startup_trials : number of completed trials before any pruning     (default 5)
        n_warmup_steps   : number of epochs before a trial can be pruned     (default 5)
    """

    #---------------------------------------------
    if pruning:
        pruner = optuna.pruners.MedianPruner(n_startup_trials = tune_config.get("pruner_startup_trials", 5),
                                             n_warmup_steps   = tune_config.get("pruner_warmup_steps", 5))
    else :
        pruner = optuna.pruners.NopPruner()

    return pruner

#================================================================================#
def run_trial(model_cls,  
              params : dict, 
              data   : dict, 
              config : dict, 
              seed   : int, 
              device,
              name   : str,
              on_epoch_end = None) -> tuple[float, dict]:
 
    """
    Train ONE model with `params` and compute the objective on the validation set.
 
    Parameters
    ----------
    model_cls    : (type[BaseDetector]) model class
    params       : (dict) complete hyperparameters config
    data         : (dict) prepared data from `data.prepae()`
    config       : (dict) run configuration -- training | tuning.objective
    seed         : (int) seed of this training
    device       : (torch.device) cpu or cuda
    name         : (str) label of the trial (Metrics name)
    on_epoch_end : optional callback (epoch, model, val_loss) given to the training loop (Optuna pruning)
 
    Returns
    ----------
    (value, history) : (tuple) objective value on the validation set | training history
    """
 
    #---------------------------------------------
    objective      = config["tuning"]["objective"]
    (X_val, y_val) = data["validation"]
 
    (model, history) = train_model(model_cls    = model_cls,
                                   params       = params,
                                   data         = data,
                                   train_config = config["training"],
                                   seed         = seed,
                                   device       = device,
                                   verbose      = True,
                                   on_epoch_end = on_epoch_end)
 
    #--------------------
    if objective == "val_loss":
        value = history["best_val_loss"]
 
    else:
        metrics = Metrics(y_true = y_val,
                          y_pred = predict_scores(model, X_val, device),
                          name   = name)
 
        value = metrics.objective(name=objective)
 
    return (float(value), history)
 
#================================================================================#
def suggest(trial, name : str, space):
 
    """
    Draw ONE hyperparameter value with Optuna.
 
        [values]                         -> trial.suggest_categorical
        {"low", "high"} (int bounds)     -> trial.suggest_int     (optional "log": true)
        {"low", "high"} (float bounds)   -> trial.suggest_float   (optional "log": true)
    """
 
    #---------------------------------------------
    if isinstance(space, dict):
 
        (low, high, log) = (space["low"], space["high"], space.get("log", False))
 
        if isinstance(low, int) and isinstance(high, int):
            return trial.suggest_int(name, low, high, log=log)
 
        return trial.suggest_float(name, float(low), float(high), log=log)
 
    #--------------------
    return trial.suggest_categorical(name, list(space))
 
#================================================================================#
def trials_DF(study : optuna.Study) -> pd.DataFrame:
 
    """
    Return a Dataframe describing the `study`
    One row per trial : number, state, value, hyperparameters, best epoch, duration

    Args
    ----------
    study : (optuna.Study) study to save as a pd.Dataframe

    Returns
    ----------
    trials : (pd.DataFrame) `study` converted as a pd.Dataframe 
    """
 
    #---------------------------------------------
    trials = study.trials_dataframe(attrs=("number", "state", "value", "params", "user_attrs", "duration"))
    trials = trials.rename(columns=lambda c: c.replace("params_", "").replace("user_attrs_", ""))
    trials = trials.rename(columns={"number": "trial"})
 
    return trials