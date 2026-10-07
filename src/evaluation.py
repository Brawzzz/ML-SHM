#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import numpy as np
import sklearn.metrics as skm

import metrics


#============================================================================================================================#
#--------------------------------------------------------- FUNCTION ---------------------------------------------------------#
#============================================================================================================================#
def select_threshold(healthy_scores: np.ndarray, method: str = "mean_std", value: float = 3.0) -> float:

    """
    Alert threshold computed on the HEALTHY VALIDATION scores only (never on test).

    Parameters
    ----------
    healthy_scores : anomaly scores of the healthy validation signals
    method         : "mean_std" -> mean + value * std  (previous behaviour, value = 3)
                     "quantile" -> value-quantile of the healthy scores (e.g. 0.99 -> ~1 % false alarms)
    value          : k for mean_std, q for quantile

    Returns
    ----------
    threshold : float
    """

    #---------------------------------------------
    healthy_scores = np.asarray(healthy_scores, dtype=float)

    if len(healthy_scores) == 0:
        raise ValueError("no healthy validation signal to compute the threshold")

    if method == "mean_std":
        return float(np.mean(healthy_scores) + value * np.std(healthy_scores))

    if method == "quantile":
        return float(np.quantile(healthy_scores, value))

    raise ValueError(f"unknown threshold method : {method} (expected 'mean_std' or 'quantile')")

#================================================================================#
def objective_value(objective: str, y: np.ndarray, scores: np.ndarray) -> float:

    """
    Ranking metric on a labelled set (healthy = 0, damaged = 1)

    objective : "pr_auc" (average precision) or "roc_auc"
    """

    #---------------------------------------------
    if len(np.unique(y)) < 2:
        raise ValueError(f"'{objective}' needs healthy AND damaged signals in the validation set "
                         f"(increase split.damaged_val_size or use objective 'val_loss')")

    if objective == "pr_auc":
        return float(skm.average_precision_score(y, scores))

    if objective == "roc_auc":
        return float(skm.roc_auc_score(y, scores))

    raise ValueError(f"unknown objective : {objective}")

#================================================================================#
def evaluate_scores(y: np.ndarray, scores: np.ndarray, threshold: float, name: str = None) -> dict:

    """
    Every metric of the model on one set : PR-AUC, ROC-AUC + metrics at the threshold

    Returns
    ----------
    dict : {"PR-AUC", "ROC-AUC", "accuracy", "precision", "recall", "f1", "f_pr", "TP", "FP", "FN", "TN"}
           (the AUCs are NaN when the set contains a single class)
    """

    #---------------------------------------------
    m = metrics.Metrics(y_true=y, y_pred=scores, name=name)

    if len(np.unique(y)) < 2:
        out = {"PR-AUC": float("nan"), "ROC-AUC": float("nan")}
        out.update({k: v for k, v in m.at_threshold(threshold).items()})
    else:
        out = m.summary(thr=threshold)

    return {k: float(v) for k, v in out.items()}

#================================================================================#
def aggregate(results: list[dict]) -> dict:

    """
    Mean and standard deviation of each metric over several runs (seeds)

    Returns
    ----------
    dict : {metric: {"mean": float, "std": float}}
    """

    #---------------------------------------------
    keys = results[0].keys()

    return {k: {"mean": float(np.nanmean([r[k] for r in results])),
                "std" : float(np.nanstd([r[k] for r in results]))} for k in keys}
