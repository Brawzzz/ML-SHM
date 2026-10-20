"""
This module describes a RUN : the complete pipeline that trains, tunes and evaluates
ONE model on ONE dataset

    -> run()            : full pipeline of a run -- data, split, scaler, tuning,
                          final trainings over several seeds, summary of the results

    -> evaluate_model() : threshold selection on the healthy validation scores,
                          then metrics on the validation and test sets

    -> save_run()       : saves results generate by a seed : model, scaler, scores, history, metrics

    -> load_trained()   : loads a model and its scaler saved by save_run()

    -> append_summary() : adds one row per run to <output_dir>/summary.csv -- fo models comparison

    -> print_summary()  : prints the test metrics of a run

Output layout of a run :

    <output_dir>/
      |  summary.csv                 
      |- <run name>/
            config.json             complete configuration used
            best_params.json        tuned hyperparameters (if tuning)
            trials.csv              every tuning trial (if tuning)
            summary.json            mean ± std of the metrics
            seed_<k>/               model.pt, scaler.pkl, scores.npz, history.json, metrics.json
"""

#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import os
import pickle
import hashlib
import json

import numpy as np
import pandas as pd
import torch

from datetime import datetime

import data
import config as cfg
import tools

from models import BaseDetector
from Metrics import Metrics

from models import get_model
from training import train_model, predict_scores
from tuning import tune
from Metrics import select_threshold, summarize_seeds
from plots import model_report


#============================================================================================================================#
#--------------------------------------------------------- FUNCTION ---------------------------------------------------------#
#============================================================================================================================#
def run(config     : dict, 
        do_tune    : bool = True, 
        do_train   : bool = True) -> dict:

    """
    Full pipeline of a RUN (model, dataset)  :

        1 -> load raw data 
        2 -> split into Train, Validation and Test sets 
        3 -> scaler fitted on Train
        4 -> tuning on validation set (if do_tune) 
        5 -> final trainings (if do_train) 
        6 -> threshold on healthy val
        7 -> evaluation on Test set 

    Args
    ----------
    config      : (dict) run configuration from config.load_run()
    do_tune     : (bool) run the hyperparameter search (if tuning.enabled) -- defaults : True
    do_train    : (bool) run the final trainings + evaluation -- defaults : True

    Returns
    ----------
    summary : summarized test metrics (mean / std over seeds), or {} if do_train is False
    """

    #---------------------------------------------
    tools.set_seed(config["seed"])

    output_dir = os.path.join(config["output_dir"], config["name"])
    device     = tools.get_device(config["device"])

    #--------------------
    os.makedirs(output_dir, exist_ok=True)
    tools.save_json(config, os.path.join(output_dir, "config.json"))

    #--------------------
    cfg.print_config(config)
    print(f" -> Training on : {device} \n")

    #-------------------------------------- DATA ---------------------------------------# 
    ds_path = os.path.join(config["data"]["dir_path"], config["data"]["dataset"]) 

    ds = data.load_dataset(name     = config["data"]["dataset"], 
                           dir_path = ds_path, 
                           **config["data"]["params"])

    #-------------------- SPLIT --------------------#
    splits       = None 
    split_config = config["split"]

    data_key   = hashlib.md5(json.dumps(config["data"]["params"], sort_keys=True).encode()).hexdigest()[:8]
    split_path = os.path.join(config["output_dir"], 
                              "splits",
                              f"{ds.name}_{data_key}_"
                              f"{split_config['method']}_"
                              f"train_{split_config['train_ratio']}_"
                              f"val_{split_config['validation_ratio']}_"
                              f"seed_{config['seed']}.npz")

    #--------------------
    if os.path.exists(split_path):

        try:
            splits = data.load_splits(split_path, ds)
        except ValueError as e:
            print(f" -> {e} : recomputed")

    #--------------------
    if splits is None:

        splits = data.split(ds          = ds, 
                            train_ratio = split_config["train_ratio"], 
                            val_ratio   = split_config["validation_ratio"], 
                            seed        = config["seed"],
                            method      = split_config["method"])
        
        splits.save(split_path, ds)

    #--------------------
    print(f"\n{splits.describe(ds)}")

    (prep_data, scaler) = data.prepare(ds, splits, config["preprocessing"]["scaler"])

    #------------------------------- HYPERPARMETERS ------------------------------------# 
    model_cls   = get_model(config["model"]["name"])
    params_path = os.path.join(output_dir, "best_params.json")

    #--------------------
    if do_tune and config["tuning"]["enabled"]:

        params = tune(model_cls = model_cls, 
                      data      = prep_data, 
                      config    = config, 
                      out_dir   = output_dir, 
                      device    = device)

    elif os.path.exists(params_path):
        params = model_cls.resolve_params(tools.load_json(params_path)["params"])
        print(f" -> Tuned hyperparameters loaded : {params_path}")

    else:
        params = model_cls.resolve_params(config["model"].get("params", {}))
        print(" -> no tuning : default hyperparameters and model.params")

    cfg.print_HP(params)

    #--------------------
    if not do_train:
        return {}

    #------------------------------- FINAL TRAIN & EVAL --------------------------------#
    runs = []
    for k in range(config["evaluation"]["n_seeds"]):

        seed_k    = config["seed"] + k
        run_dir_k = os.path.join(output_dir, f"seed_{k}" )

        (model_k, history_k) = train_model(model_cls    = model_cls, 
                                           params       = params, 
                                           data         = prep_data, 
                                           train_config = config["training"], 
                                           seed         = seed_k, 
                                           device       = device)

        (results_k, scores_k) = evaluate_model(model            = model_k, 
                                               data             = prep_data, 
                                               threshold_config = config["threshold"], 
                                               device           = device,
                                               name             = f"{config['name']}_seed{k}")

        save_run(run_dir = run_dir_k, 
                 model   = model_k, 
                 params  = params, 
                 scaler  = scaler, 
                 history = history_k, 
                 results = results_k, 
                 scores  = scores_k, 
                 data    = prep_data)

        runs.append(results_k)

        print(f" -> seed {seed_k} : Test : "
              f"PR-AUC = {results_k['test']['PR-AUC']:.4f} | "
              f"ROC-AUC = {results_k['test']['ROC-AUC']:.4f} | "
              f"F1 = {results_k['test']['f1']:.4f} | "
              f"FPR = {results_k['test']['f_pr']:.4f}\n")

    #---------------------------------------------
    summary = {"test"       : summarize_seeds([r["test"] for r in runs]),
               "validation" : summarize_seeds([r["validation"]  for r in runs]),
               "params"     : params}

    append_summary(config, summary, splits_desc=splits.describe(ds))
    print_summary(config, summary)

    tools.save_json(summary, os.path.join(output_dir, "summary.json"))

    return summary

#================================================================================#
def evaluate_model(model            : BaseDetector, 
                   data             : dict, 
                   threshold_config : dict, 
                   device, 
                   name             : str) -> tuple[dict, dict]:

    """
    Model evaluation :
    
        -> threshold selected on healthy validation scores
        -> metrics computed on Validation and Test sets

    Args
    ----------
    model            : (BaseDetector) trained model to evaluate
    data             : (dict) dict containing the prepared data
    threshold_config : (dict) configuration for threshold computing
    device           : (torch.device) cpu or cuda
    name             : (str) label of the model, used by Metrics in prints and file names

    Returns
    ----------
    (results, scores) : (tuple) metric and scores computed on Validation and Test sets
    """

    #---------------------------------------------
    (X_val,  y_val)  = data["validation"]
    (X_test, y_test) = data["test"]

    scores = {"validation" : predict_scores(model, X_val,  device), 
              "test"       : predict_scores(model, X_test, device)}

    threshold = select_threshold(scores["validation"], y_val, **threshold_config)

    results = {"threshold"  : threshold,
               "validation" : Metrics(y_val,  scores["validation"], name).summary(thr=threshold),
               "test"       : Metrics(y_test, scores["test"],       name).summary(thr=threshold)}

    return (results, scores)

#================================================================================#
def save_run(run_dir    : str, 
             model      : BaseDetector, 
             params     : dict, 
             scaler, 
             history    : dict, 
             results    : dict, 
             scores     : dict, 
             data       : dict) -> None:

    """
    Save everything needed to reload the model and redraw the figures without retraining
    
    Args
    ----------
    run_dir : (str) output directory of this seed (created if not exists)
    model   : (BaseDetector) trained model
    params  : (dict) complete hyperparameters used to build the model
    scaler  : fitted scaler (needed to normalise new signals the same way)
    history : (dict) training history returned by train_model
    results : (dict) metrics returned by evaluate_model
    scores  : (dict) anomaly scores returned by evaluate_model
    data    : (dict) prepared data (only the validation and test labels are saved)

    Returns
    ----------
    None
    """

    #---------------------------------------------
    os.makedirs(run_dir, exist_ok=True)

    threshold = results["threshold"]

    (_, y_val)  = data["validation"]
    (_, y_test) = data["test"]

    #--------------------
    torch.save({"model"        : model.NAME,
                "params"       : params,
                "input_length" : model.input_length,
                "in_channels" : model.in_channels,
                "state_dict"   : model.state_dict()}, os.path.join(run_dir, "model.pt"))

    with open(os.path.join(run_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    np.savez_compressed(file        = os.path.join(run_dir, "scores.npz"),
                        val_scores  = scores["validation"],
                        val_y       = y_val,
                        test_scores = scores["test"], 
                        test_y      = y_test,
                        threshold   = threshold)

    tools.save_json(history, os.path.join(run_dir, "history.json"))
    tools.save_json(results, os.path.join(run_dir, "metrics.json"))

#================================================================================#
def plot_run(run_dir : str, 
             seed    : int  = 0, 
             show    : bool = True) -> None:

    """
    Report of a trained run, rebuilt from its saved files

    Reads the files written by save_run() in <run_dir>/seed_<seed>/ :
        - scores.npz   : test labels, test anomaly scores and alert threshold
        - history.json : training and validation loss of each epoch

    Prints the test metrics and draws the report figure with plots.model_report() :

    Args
    ----------
    run_dir : (str) run directory -- e.g. results/CAE_UTAH_pr_auc
    seed    : (int) index of the final training to plot (folder seed_<seed>) -- defaults : 0
    show    : (bool) whether to display the figure (False : only save it) -- defaults : True

    Returns
    ----------
    None

    Example
    ----------
    plot_run("results/CAE_UTAH_pr_auc", seed=1, show=False)
    """

    #---------------------------------------------
    seed_dir = os.path.join(run_dir, f"seed_{seed}")
    scores   = np.load(os.path.join(seed_dir, "scores.npz"))
    history  = tools.load_json(os.path.join(seed_dir, "history.json"))
    thr      = float(scores["threshold"])

    metrics = Metrics(y_true = scores["test_y"], 
                      y_pred = scores["test_scores"],
                      name   = f"{os.path.basename(os.path.normpath(run_dir))}_seed_{seed}")
    
    metrics.summary(thr=thr, show=True)

    model_report(metrics      = metrics, 
                 thr          = thr, 
                 train_losses = history["train_loss"], 
                 val_losses   = history["val_loss"], 
                 save_dir     = run_dir, 
                 show         = show)

#================================================================================#
def load_trained(run_dir: str, device="cpu") -> tuple[BaseDetector, object]:

    """
    Loads a trained model and its scaler from a seed directory written by save_run().

    The model is rebuilt from the saved hyperparameters (same architecture),
    then the trained weights are loaded into it. 

    Args
    ----------
    run_dir : (str) seed directory containing model.pt and scaler.pkl
              (e.g. ./results/CAE_UTAH_pr_auc/seed_0)

    device  : (str | torch.device) device on which the weights are loaded -- defaults : "cpu"

    Returns
    ----------
    (model, scaler) : (tuple)
                      model  -> (BaseDetector) trained model
                      scaler -> scaler fitted on Train set
    """

    #---------------------------------------------
    checkpoint  = torch.load(os.path.join(run_dir, "model.pt"), map_location=device, weights_only=False)

    model = get_model(checkpoint["model"]).build(checkpoint["input_length"], checkpoint["params"], in_channels=checkpoint.get("in_channels", 1))
    
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with open(os.path.join(run_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    return (model, scaler)

#================================================================================#
def append_summary(config: dict, summary: dict, splits_desc: str) -> None:

    """
    Add one row describing a run to <output_dir>/summary.csv

    The row contains the run parameters : 
        date, name, dataset, model, split, tuning objective, threshold, number of seeds
    The test metrics : mean and std over the seeds
    The hyperparameters used and the size of each set.

    Args
    ----------
    config      : (dict) run configuration
    summary     : (dict) aggregated results built in run() :
                  {"test": {metric: {"mean", "std"}}, "validation": {...}, "params": dict}
    splits_desc : (str) description of the split (Splits.describe), saved with the row

    Returns
    ----------
    None
    """

    #---------------------------------------------
    row = {"date"       : datetime.now().strftime("%Y-%m-%d %H:%M"),
           "experiment" : config["name"],
           "dataset"    : config["data"]["dataset"],
           "model"      : config["model"]["name"],
           "split"      : f"{config['split']['method']} | train = {config['split']['train_ratio']} | validation = {config['split']['validation_ratio']}",
           "objective"  : config["tuning"]["objective"] if config["tuning"]["enabled"] else "none",
           "threshold"  : f"{config['threshold']['method']}({config['threshold']['value']})",
           "n_seeds"    : config["evaluation"]["n_seeds"]}

    for metric in ("PR-AUC", "ROC-AUC", "f1", "precision", "recall", "f_pr"):

        row[f"test_{metric}_mean"] = summary["test"][metric]["mean"]
        row[f"test_{metric}_std"]  = summary["test"][metric]["std"]

    row["params"] = str(summary["params"])
    row["sizes"]  = splits_desc

    #---------------------------------------------
    path = os.path.join(config["output_dir"], "summary.csv")
    df   = pd.DataFrame([row])

    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)

    df.to_csv(path, index=False)
    print(f" -> results saved in {path}")

#================================================================================#
def print_summary(config : dict, summary : dict) -> None:

    """
    Print the test metrics of a run, as mean ± std over the seeds.

    Args
    ----------
    config  : (dict) run configuration (name and evaluation.n_seeds are used)
    summary : (dict) aggregated results built in run()

    Returns
    ----------
    None

    Example
    ----------
        #--------------- RESULTS CAE_UTAH_pr_auc (Test, 3 seed(s)) ---------------#
          PR-AUC    : 0.9312 ± 0.0081
          ROC-AUC   : 0.9655 ± 0.0042
          f1        : 0.8400 ± 0.0220
          ...
    """

    print(f"\n#--------------- RESULTS {config['name']} (Test, {config['evaluation']['n_seeds']} seed(s)) ---------------#")

    for metric in ("PR-AUC", "ROC-AUC", "f1", "precision", "recall", "f_pr"):
        s = summary["test"][metric]
        print(f"  {metric:<10}: {s['mean']:.4f} ± {s['std']:.4f}")

    print("#----------------------------------------------------------------------------#\n")