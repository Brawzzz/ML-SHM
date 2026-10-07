#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import os
import optuna

import tools

from models import BaseDetector
from training import fit, predict_scores
from evaluation import objective_value


#============================================================================================================================#
#--------------------------------------------------------- CONSTANT ---------------------------------------------------------#
#============================================================================================================================#
OBJECTIVES = {
    "val_loss" : "minimize",    # unsupervised : model loss on the HEALTHY validation signals
    "pr_auc"   : "maximize",    # semi-supervised : average precision on the validation set (needs damaged signals)
    "roc_auc"  : "maximize",    # semi-supervised : ROC-AUC on the validation set (needs damaged signals)
}

#============================================================================================================================#
#--------------------------------------------------------- FUNCTION ---------------------------------------------------------#
#============================================================================================================================#
def tune(model_cls: type[BaseDetector],
         data: dict,
         cfg: dict,
         out_dir: str,
         device) -> tuple[dict, optuna.Study]:

    """
    Hyperparameter search with Optuna (TPE sampler + median pruning).

    Every trial : sample the hyperparameters from the model's SEARCH_SPACE, train on
    data["train"] with early stopping on the healthy validation loss, then score the
    restored best weights on the validation set according to the chosen objective.
    The test set is NEVER used here.

    Parameters
    ----------
    model_cls : BaseDetector subclass to tune
    data      : {"train": (X, y), "val": (X, y), ...} from data.prepare()
    cfg       : full experiment config (tuning, training, model, seed sections are used)
    out_dir   : experiment directory (trials.csv, best_params.json, optuna.db are written there)
    device    : torch device

    Returns
    ----------
    (best_params, study) : best_params is the FULL parameter dict (defaults + tuned + fixed)
    """

    #---------------------------------------------
    t_cfg     = cfg["tuning"]
    tr_cfg    = cfg["training"]
    objective = t_cfg["objective"]
    seed      = cfg["seed"]

    if objective not in OBJECTIVES:
        raise ValueError(f"unknown tuning objective : {objective} (available : {list(OBJECTIVES)})")

    (X_val, y_val) = data["val"]
    val_healthy    = (X_val[y_val == 0], y_val[y_val == 0])

    fixed        = cfg["model"].get("params", {})
    search_space = cfg["model"].get("search_space", {})

    #---------------------------------------------
    def trial_objective(trial: optuna.Trial) -> float:

        params = model_cls.suggest_params(trial, fixed=fixed, search_space=search_space)
        (arch, train_p) = model_cls.split_params(params)

        tools.set_seed(seed + trial.number)
        model = model_cls(input_length=X_val.shape[1], **arch)

        #------------------------------ pruning on the objective, epoch by epoch
        def on_epoch_end(epoch, mdl, val_loss):

            if objective == "val_loss":
                value = val_loss
            else:
                value = objective_value(objective, y_val, predict_scores(mdl, X_val, device))

            trial.report(value, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

        #------------------------------
        history = fit(model, data["train"], val_healthy,
                      **train_p,
                      epochs       = tr_cfg["epochs"],
                      patience     = tr_cfg["patience"],
                      device       = device,
                      num_workers  = tr_cfg["num_workers"],
                      seed         = seed + trial.number,
                      on_epoch_end = on_epoch_end,
                      verbose      = False)

        trial.set_user_attr("best_epoch", history["best_epoch"])
        trial.set_user_attr("best_val_loss", history["best_val_loss"])

        if objective == "val_loss":
            return history["best_val_loss"]

        return objective_value(objective, y_val, predict_scores(model, X_val, device))

    #---------------------------------------------
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    storage = None
    if t_cfg.get("resume", False):
        storage = f"sqlite:///{os.path.abspath(os.path.join(out_dir, 'optuna.db'))}"

    study = optuna.create_study(study_name     = cfg["name"],
                                direction      = OBJECTIVES[objective],
                                sampler        = optuna.samplers.TPESampler(seed=seed),
                                pruner         = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
                                storage        = storage,
                                load_if_exists = True)

    n_done   = len([t for t in study.trials if t.state.is_finished()])
    n_trials = max(0, t_cfg["n_trials"] - n_done)

    print(f"\n#--------------- TUNING {cfg['name']} : {n_trials} trial(s), objective = {objective} "
          f"({OBJECTIVES[objective]}) ---------------#\n")

    study.optimize(trial_objective,
                   n_trials          = n_trials,
                   timeout           = t_cfg.get("timeout"),
                   callbacks         = [_print_trial],
                   gc_after_trial    = True,
                   show_progress_bar = False)

    #---------------------------------------------
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("no trial completed : check the search space / training settings")

    best_params = model_cls.resolve_params({**study.best_params, **fixed})

    study.trials_dataframe().to_csv(os.path.join(out_dir, "trials.csv"), index=False)
    tools.save_json({"objective"   : objective,
                     "best_value"  : study.best_value,
                     "best_trial"  : study.best_trial.number,
                     "n_trials"    : len(study.trials),
                     "params"      : best_params}, os.path.join(out_dir, "best_params.json"))

    print(f"\n -> best {objective} : {study.best_value:.4g} (trial {study.best_trial.number})")
    print(f" -> best params : {best_params}\n")

    return (best_params, study)

#================================================================================#
def _print_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:

    state = trial.state.name.lower()
    value = f"{trial.value:.4g}" if trial.value is not None else "-"

    try:
        best = f"{study.best_value:.4g}"
    except ValueError:
        best = "-"

    print(f" trial {trial.number:>3} [{state:<8}] value = {value:<10} best = {best:<10} {trial.params}")
