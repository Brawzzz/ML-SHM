#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import os
import numpy as np
import pandas as pd
import sklearn.metrics as skm


#============================================================================================================================#
#---------------------------------------------------------- CLASS -----------------------------------------------------------#
#============================================================================================================================#
class Metrics:

    """
    Metrics to compare models from damage state : damaged = positive = 1

    Attributes
    ----------
    y_true  : (np.ndarray) labels -> 0 for healthy | 1 for damaged
    y_score : (np.ndarray) predicted scores or probabilities for the positive class (damaged).
    name    : (str) mame of the model or metric set for identification in plots and reports
    """

    #================================================================================#
    def __init__(self, y_true : np.ndarray, y_pred : np.ndarray, name : str):

        self.name = name

        self.y_true  = np.asarray(y_true)
        self.y_score  = np.asarray(y_pred) 

    #================================================================================#
    @classmethod
    def from_mse(cls, healthy_mse, crack_mse, name: str = None) -> "Metrics":

        y_true  = np.concatenate([np.zeros(len(healthy_mse)), np.ones(len(crack_mse))])
        y_score = np.concatenate([healthy_mse, crack_mse])

        return cls(y_true, y_score, name=name)

    #================================================================================#
    def _mixed_signals(self):
        
        """
        return true if the set contains healthy AND damaged signals
        """
        #---------------------------------------------
        return len(np.unique(self.y_true)) == 2
    
    #================================================================================#
    def objective(self, name: str) -> float:

        """
        Value to optimise by the hyperparameter search : "pr_auc" or "roc_auc"
        (computed exclusively on validation set)

        Args:
        ----------
        name : (str) name of the value to optimize

        Returns
        ----------
        (float) : computed criterion value
        """

        #---------------------------------------------
        if not self._mixed_signals() :
            raise ValueError(f"'{name}' needs healthy AND damaged signals "
                             f"(increase the damaged ratio in validation or use the objective 'val_loss')")

        if name == "pr_auc":
            return self.PR_auc()

        if name == "roc_auc":
            return self.ROC_auc()

        raise ValueError(f"unknown objective : {name} (expected 'pr_auc' or 'roc_auc')")
    
    #================================================================================#
    def PR_auc(self) -> float:   

        """
        Compute the Area Under the Curve (AUC) for the Precision-Recall (PR) curve.

        Returns
        ----------
        pr_auc : Area Under the Curve (AUC) score for the PR curve.
        """
        #---------------------------------------------
        return skm.average_precision_score(self.y_true, self.y_score)

    #================================================================================#
    def ROC_auc(self) -> float:  

        """
        Compute the Area Under the Curve (AUC) for the Receiver Operating Characteristic (ROC) curve.

        Returns
        ----------
        roc_auc : Area Under the Curve (AUC) score for the ROC curve.
        """

        #---------------------------------------------
        return skm.roc_auc_score(self.y_true, self.y_score)

    #================================================================================#
    def at_threshold(self, thr : float) -> dict:

        """
        Compute different metrics for a model at a specific threshold

        Args
        ----------
        thr : (float) threshold value for classification.
        
        Returns
        ----------
        dict : Dictionary containing : 
                - precision
                - recall
                - f1-score
                - false positive rate (f_pr)
        """

        #---------------------------------------------
        CM = self.confusion(thr=thr)

        TP = CM[1, 1]
        FP = CM[0, 1]
        FN = CM[1, 0]
        TN = CM[0, 0]

        accuracy  = _accuracy(TP, FP, TN, FN)
        precision = _precision(TP, FP)
        recall    = _recall(TP, FN)
        f_pr      = _fpr(FP, TN)
        f1        = _Fscore(precision, recall)

        metrics = {
            "accuracy": accuracy,
            "precision": precision, 
            "recall": recall, 
            "f1": f1, 
            "f_pr": f_pr,
            "TP" : TP, 
            "FP" : FP,
            "FN" : FN,
            "TN" : TN
            }
        
        return metrics
    
    #================================================================================#
    def confusion(self, thr : float) -> np.ndarray:

        """
        Compute the confusion matrix on a specific threshold.

        Args
        ----------
        thr : (float) threshold value for classification.

        Returns
        ----------
        CM : (np.ndarray) confusion matrix as a 2D array.
        """

        #---------------------------------------------
        y_pred  = (self.y_score > thr).astype(int)
        CM      = skm.confusion_matrix(self.y_true, y_pred, labels=[0, 1])

        return CM

    #================================================================================#
    def summary(self, 
                thr      : float = None, 
                show     : bool  = False, 
                save_dir : str   = None) -> dict:
 
        """
        Summarize all the metrics of a model in a dictionary.
 
        Args
        ----------
        thr      : (float) Optional threshold value for classification. If provided, additional metrics will be included.
        show     : (bool) whether to print the metrics -- defaults False
        save_dir : (bool) directory where to save the metrics as a csv file -- defaults ./results -- if None no save

        Returns
        ----------
        dict : Dictionary containing the following metrics:
                - PR-AUC
                - ROC-AUC
                - precision (if thr provided)
                - recall (if thr provided)
                - f1-score (if thr provided)
                - false positive rate (f_pr) (if thr provided)
        """

        #---------------------------------------------
        if self._mixed_signals():
            summary_dict = {"PR-AUC": self.PR_auc(), "ROC-AUC": self.ROC_auc()}

        else:
            summary_dict = {"PR-AUC": float("nan"), "ROC-AUC": float("nan")}

        #--------------------
        if thr is not None:

            pt = self.at_threshold(thr)

            summary_dict.update({
                "accuracy": pt["accuracy"],
                "precision": pt["precision"], 
                "recall": pt["recall"], 
                "f1": pt["f1"], 
                "f_pr": pt["f_pr"],
                "TP" : pt["TP"], 
                "FP" : pt["FP"],
                "FN" : pt["FN"],
                "TN" : pt["TN"]
                })

        #-------------------------
        if show:
            print(f"\n#--------- Metrics : {self.name or 'model'} ---------#")

            for k, v in summary_dict.items():
                print(f"  {k:<10}: {v:.4f}")

            if thr is not None:
                print(f"  (threshold = {thr:.2e})")
                
            print("#-----------------------------------------------#\n")

        #-------------------------
        if save_dir is not None:

            os.makedirs(save_dir, exist_ok=True)

            thr_tag   = f"_at_{thr:.5f}" if thr is not None else ""
            save_path = os.path.join(save_dir, f"{self.name}_metrics{thr_tag}.csv")

            pd.DataFrame(summary_dict, index=[0]).to_csv(save_path, index=False)
            print(f"Metrics saved at : {save_path}")

        return summary_dict

#============================================================================================================================#
#------------------------------------------------------ FUNCTIONS -----------------------------------------------------------#
#============================================================================================================================#
def _accuracy(TP : float, FP : float,
              TN : float, FN : float) -> float:

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) else 0.0
    return accuracy

#================================================================================#
def _precision(TP : float, FP : float) -> float :

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    return precision

#================================================================================#
def _recall(TP : float, FN : float) -> float :

    recall = TP / (TP + FN) if (TP + FN) else 0.0
    return recall

#================================================================================#
def _Fscore(precision : float, recall : float) -> float :

    f_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return f_score

#================================================================================#
def _fpr(FP : float, TN : float):

    fpr = FP / (FP + TN) if (FP + TN) else 0.0
    return fpr

#================================================================================#
def select_threshold(scores : np.ndarray,
                     y      : np.ndarray = None, 
                     method : str        = "mean_std", 
                     value  : float      = 3.0) -> float:

    """
    Computes the alert threshold on the healthy validation scores only.
    Computation depends on the given `method`.

    Args
    ----------
    healthy_scores : (np.ndarray) anomaly scores of the healthy validation signals
    method         : (str) determines the method to compute the threshold alert 
                        "mean_std" -> mean + value * std 
                        "quantile" -> value-quantile of the healthy scores 
    value          : (float) coefficiant depending on `method`
                        "mean_std" -> k :
                        "quantile" -> q :

                    Compromise :     
                        value plus petit  →  seuil plus bas   →  rappel ↑   fausses alarmes ↑
                        value plus grand  →  seuil plus haut  →  rappel ↓   fausses alarmes ↓

    Returns
    ----------
    threshold : (float) computed threshold based on the given `method`
    """

    #---------------------------------------------
    scores         = np.asarray(scores, dtype=float)
    healthy_scores = scores if y is None else scores[np.asarray(y) == 0]

    if len(healthy_scores) == 0:
        raise ValueError("no healthy validation signal to compute the threshold")

    #--------------------
    if method == "mean_std":

        threshold = float(np.mean(healthy_scores) + value * np.std(healthy_scores))
        return threshold

    #--------------------
    if method == "quantile":

        threshold = float(np.quantile(healthy_scores, value))
        return threshold

    #--------------------
    if method == "youden":

        if y is None or len(np.unique(y)) < 2:
            raise ValueError("'youden' needs healthy AND damaged validation signals")
        
        (fpr, tpr, thr) = skm.roc_curve(y, scores)

        return float(thr[np.argmax(tpr - fpr)])

    #--------------------
    raise ValueError(f"unknown threshold method : {method} (expected 'mean_std' or 'quantile')")

#================================================================================#
def summarize_seeds(results: list[dict]) -> dict:

    """
    Returns computed mean and standard deviation of each metric over several trainings (seeds).
    NaN values are ignored (e.g. AUC on a single-class set)

    Args
    ----------
    results : (list[dict]) metrics of several trainings

    Returns
    ----------
    dict : {metric: {"mean": float, "std": float}}
    """

    #---------------------------------------------
    return {key: {"mean": float(np.nanmean([r[key] for r in results])),
                  "std" : float(np.nanstd([r[key] for r in results]))} for key in results[0]}