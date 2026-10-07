#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import os
import pickle

from datetime import datetime

import numpy as np
import pandas as pd
import torch

import data
import tools

from config import print_config
from models import get_model, BaseDetector
from training import fit, predict_scores
from tuning import tune
from evaluation import select_threshold, evaluate_scores, aggregate


#============================================================================================================================#
#--------------------------------------------------------- FUNCTION ---------------------------------------------------------#
#============================================================================================================================#
def run_experiment(cfg: dict, do_tune: bool = True, do_train: bool = True) -> dict:

    """
    Full pipeline of one (model, dataset) experiment :

        load raw data -> split (shared by every model) -> scaler fitted on train
        -> [tuning on val] -> final trainings (n_seeds) -> threshold on healthy val
        -> evaluation on test -> results/<name>/ + one row in results/summary.csv

    Parameters
    ----------
    cfg      : experiment config from config.load_experiment()
    do_tune  : run the hyperparameter search (if tuning.enabled)
    do_train : run the final trainings + evaluation

    Returns
    ----------
    summary : aggregated test metrics (mean / std over seeds), or {} if do_train is False
    """

    #---------------------------------------------
    out_dir = os.path.join(cfg["output_dir"], cfg["name"])
    os.makedirs(out_dir, exist_ok=True)

    print_config(cfg)
    tools.save_json(cfg, os.path.join(out_dir, "config.json"))

    tools.set_seed(cfg["seed"])
    device = tools.get_device(cfg["device"])
    print(f" -> device : {device}")

    #--------------------------------------------- data
    ds = data.load_dataset(cfg["data"]["dataset"], cfg["data"]["dir_path"], **cfg["data"]["params"])

    splits = data.load_or_make_splits(ds, cfg["split"], cfg["data"], cfg["seed"],
                                      splits_dir=os.path.join(cfg["output_dir"], "splits"))
    print(f" -> {splits.describe(ds)}\n")

    (prepared, scaler) = data.prepare(ds, splits, cfg["preprocessing"]["scaler"])

    #--------------------------------------------- hyperparameters
    model_cls   = get_model(cfg["model"]["type"])
    params_path = os.path.join(out_dir, "best_params.json")

    if do_tune and cfg["tuning"]["enabled"]:
        (params, _) = tune(model_cls, prepared, cfg, out_dir, device)

    elif os.path.exists(params_path):
        params = model_cls.resolve_params(tools.load_json(params_path)["params"])
        print(f" -> tuned hyperparameters loaded : {params_path}")

    else:
        params = model_cls.resolve_params(cfg["model"].get("params", {}))
        print(" -> no tuning : default hyperparameters (+ model.params)")

    print(f" -> hyperparameters : {params}\n")

    if not do_train:
        return {}

    #--------------------------------------------- final trainings + evaluation
    runs = []
    for k in range(cfg["evaluation"]["n_seeds"]):
        runs.append(train_and_evaluate(model_cls, params, prepared, scaler, cfg,
                                       seed=cfg["seed"] + k,
                                       run_dir=os.path.join(out_dir, f"seed_{k}"),
                                       device=device))

    #---------------------------------------------
    summary = {"test" : aggregate([r["test"] for r in runs]),
               "val"  : aggregate([r["val"]  for r in runs]),
               "params" : params}

    tools.save_json(summary, os.path.join(out_dir, "summary.json"))
    _append_summary(cfg, summary, splits_desc=splits.describe(ds))
    _print_summary(cfg, summary)

    return summary

#================================================================================#
def train_and_evaluate(model_cls: type[BaseDetector], params: dict, prepared: dict, scaler,
                       cfg: dict, seed: int, run_dir: str, device) -> dict:

    """
    Train one model with fixed hyperparameters, choose the threshold on the healthy
    validation scores, evaluate on val and test, and save everything in run_dir.
    """

    #---------------------------------------------
    os.makedirs(run_dir, exist_ok=True)
    tools.set_seed(seed)

    (X_tr, y_tr)   = prepared["train"]
    (X_val, y_val) = prepared["val"]
    (X_te, y_te)   = prepared["test"]

    (arch, train_p) = model_cls.split_params(params)
    model = model_cls(input_length=X_tr.shape[1], **arch)

    #---------------------------------------------
    history = fit(model, (X_tr, y_tr), (X_val[y_val == 0], y_val[y_val == 0]),
                  **train_p,
                  epochs      = cfg["training"]["epochs"],
                  patience    = cfg["training"]["patience"],
                  device      = device,
                  num_workers = cfg["training"]["num_workers"],
                  seed        = seed)

    #---------------------------------------------
    val_scores  = predict_scores(model, X_val, device)
    test_scores = predict_scores(model, X_te, device)

    threshold = select_threshold(val_scores[y_val == 0], **cfg["threshold"])

    results = {"seed"      : seed,
               "threshold" : threshold,
               "val"       : evaluate_scores(y_val, val_scores, threshold),
               "test"      : evaluate_scores(y_te, test_scores, threshold)}

    #--------------------------------------------- artifacts
    torch.save({"model"        : model_cls.NAME,
                "params"       : params,
                "input_length" : X_tr.shape[1],
                "state_dict"   : model.state_dict()}, os.path.join(run_dir, "model.pt"))

    with open(os.path.join(run_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    np.savez_compressed(os.path.join(run_dir, "scores.npz"),
                        val_scores=val_scores, val_y=y_val,
                        test_scores=test_scores, test_y=y_te,
                        threshold=threshold)

    tools.save_json(history, os.path.join(run_dir, "history.json"))
    tools.save_json(results, os.path.join(run_dir, "metrics.json"))

    print(f" -> seed {seed} : test PR-AUC = {results['test']['PR-AUC']:.4f} | "
          f"ROC-AUC = {results['test']['ROC-AUC']:.4f} | F1 = {results['test']['f1']:.4f} | "
          f"FPR = {results['test']['f_pr']:.4f}\n")

    return results

#================================================================================#
def load_trained(run_dir: str, device="cpu") -> tuple[BaseDetector, object]:

    """
    Reload a model trained by train_and_evaluate() and its scaler

    Returns
    ----------
    (model, scaler)
    """

    #---------------------------------------------
    ckpt  = torch.load(os.path.join(run_dir, "model.pt"), map_location=device, weights_only=False)
    model = get_model(ckpt["model"]).build(ckpt["input_length"], ckpt["params"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with open(os.path.join(run_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    return (model, scaler)

#================================================================================#
def _append_summary(cfg: dict, summary: dict, splits_desc: str) -> None:

    """One row per experiment in <output_dir>/summary.csv : the table used to compare models"""

    #---------------------------------------------
    row = {"date"       : datetime.now().strftime("%Y-%m-%d %H:%M"),
           "experiment" : cfg["name"],
           "dataset"    : cfg["data"]["dataset"],
           "model"      : cfg["model"]["type"],
           "split"      : cfg["split"]["strategy"],
           "objective"  : cfg["tuning"]["objective"] if cfg["tuning"]["enabled"] else "none",
           "threshold"  : f"{cfg['threshold']['method']}({cfg['threshold']['value']})",
           "n_seeds"    : cfg["evaluation"]["n_seeds"]}

    for metric in ("PR-AUC", "ROC-AUC", "f1", "precision", "recall", "f_pr"):
        row[f"test_{metric}_mean"] = summary["test"][metric]["mean"]
        row[f"test_{metric}_std"]  = summary["test"][metric]["std"]

    row["params"] = str(summary["params"])
    row["sizes"]  = splits_desc

    #---------------------------------------------
    path = os.path.join(cfg["output_dir"], "summary.csv")
    df   = pd.DataFrame([row])

    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)

    df.to_csv(path, index=False)
    print(f" -> results added to {path}")

#================================================================================#
def _print_summary(cfg: dict, summary: dict) -> None:

    print(f"\n#--------------- RESULTS {cfg['name']} (test, {cfg['evaluation']['n_seeds']} seed(s)) ---------------#")

    for metric in ("PR-AUC", "ROC-AUC", "f1", "precision", "recall", "f_pr"):
        s = summary["test"][metric]
        print(f"  {metric:<10}: {s['mean']:.4f} ± {s['std']:.4f}")

    print("#----------------------------------------------#\n")
