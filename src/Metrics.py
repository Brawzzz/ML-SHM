#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as skm
import matplotlib.pyplot as plt


#============================================================================================================================#
#---------------------------------------------------------- CLASS -----------------------------------------------------------#
#============================================================================================================================#
class Metrics:

    """
    Metrics to compare models from damage state : dammaged = positive = 1

    Attributes
    ----------
    y_true : np.ndarray
        True labels (0 for healthy, 1 for damaged).

    y_score : np.ndarray
        Predicted scores or probabilities for the positive class (damaged).

    name : str
        Name of the model or metric set for identification in plots and reports.s
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
    def PR_curve(self, ax=None, show: bool = True) -> float:

        """
        plot PR - Precision-Recall curve and return AUC score

        Parameters
        ----------
        ax      : Axes to plot on -- if None, a new figure and axes will be created -- default None.
        show    : Whether to display the plot -- default True.

        Returns
        ----------
        pr_auc : Area Under the Curve (AUC) score for the PR curve.
        """

        #---------------------------------------------
        pr_auc                 = self.PR_auc()
        (precision, recall, _) = skm.precision_recall_curve(self.y_true, self.y_score)

        #-------------------------
        base = self.y_true.mean()

        if ax is None:
            label   = f"Baseline ({base:.3f})"
            (_, ax) = plt.subplots(figsize=(7, 6))
            sns.set_theme(style="whitegrid");
        else:
            label = None

        #-------------------------
        ax.plot(recall, precision, lw=2, label=f"{self.name or 'PR'} (AP={pr_auc:.3f})")
        ax.axhline(base, color="grey", lw=1, ls="--",label=label)

        ax.set_title("Courbe Precision-Recall")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)

        ax.legend(loc="lower left")

        #-------------------------
        if ax is None and show : 
            plt.show()

        return pr_auc

    #================================================================================#
    def ROC_curve(self, ax=None, show: bool = True) -> float:

        """
        plot ROC curve and return AUC score

        Parameters
        ----------
        ax      : Axes to plot on -- if None, a new figure and axes will be created -- default None.
        show    : Whether to display the plot -- default True.

        Returns
        ----------
        roc_auc : Area Under the Curve (AUC) score for the ROC curve.
        """

        #---------------------------------------------
        roc_auc         = self.ROC_auc()
        (f_pr, t_pr, _) = skm.roc_curve(self.y_true, self.y_score)

        #-------------------------
        if ax is None:
            label   = ("Aléatoire")
            (_, ax) = plt.subplots(figsize=(7, 6))
            sns.set_theme(style="whitegrid")
            
        else:
            label = None

        #-------------------------
        ax.plot(f_pr, t_pr, lw=2, label=f"{self.name or 'ROC'} (AUC={roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--", label=label)

        ax.set_title("Courbe ROC")
        ax.set_xlabel("FPR (fausses alarmes)")
        ax.set_ylabel("TPR (rappel)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

        ax.legend(loc="lower right")

        #-------------------------
        if ax is None and show :
            plt.show()

        return roc_auc

    #================================================================================#
    def at_threshold(self, thr : float) -> dict:

        """
        Compute different metrics for a model at a specific threshold

        Parameters
        ----------
        thr : Threshold value for classification.
        
        Returns
        ----------
        dict : Dictionary containing : 
                - precision
                - recall
                - f1-score
                - false positive rate (f_pr)
        """

        #---------------------------------------------
        CM = self.confusion(threshold=thr, plot=False)

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
    def confusion(self, threshold : float, plot : bool = False, ax=None) -> np.ndarray:

        """
        Compute and plot the confusion matrix based on the threshold.

        Parameters
        ----------
        threshold   : Threshold value for classification.
        plot        : Whether to plot the confusion matrix -- default false
        ax          : Axes object for plotting -- default None

        Returns
        ----------
        CM : Confusion matrix as a 2D array.
        """

        #---------------------------------------------
        y_pred  = (self.y_score > threshold).astype(int)
        CM      = skm.confusion_matrix(self.y_true, y_pred, labels=[0, 1])

        #-------------------------
        if plot:
            
            if ax is None:
                (_, ax) = plt.subplots(figsize=(6, 5))

            #-------------------------
            ax.set_title("Confusion Matrix", fontsize=14)
            sns.heatmap(data=CM, 
                        annot=True, 
                        fmt="d", 
                        cmap="Blues", 
                        ax=ax,
                        xticklabels=["Predicted Healthy", "Predicted Damaged"],
                        yticklabels=["Healthy", "Damaged"])
        
        return CM

    #================================================================================#
    def distribution_error(self, threshold: float, ax : plt.Axes) -> None:

        """
        MSE histogramme healty / dammaged + threshold line

        Parameters
        ----------
        threshold   : Threshold value for classification.
        ax          : Axes object for plotting.

        Returns
        ----------
        None
        """
        
        #---------------------------------------------
        healthy = self.y_score[self.y_true == 0]
        damaged = self.y_score[self.y_true == 1]
 
        sns.histplot(healthy, bins=50, color="green", alpha=0.6, label="Healthy", ax=ax, stat="density")
        sns.histplot(damaged, bins=50, color="red",   alpha=0.5, label="Damaged", ax=ax, stat="density")

        ax.axvline(threshold, color="black", linestyle="dashed", linewidth=2, label="Warning threshold")
        ax.set_title("Reconstruction error separation", fontsize=14)
        ax.set_xlabel("MSE Error"); ax.set_ylabel("Number of signal"); ax.legend()
    
    #================================================================================#
    def summary(self, thr: float = None, show : bool = False, save : bool = False) -> dict:

        """
        Summarize all the metrics of a model in a dictionary.

        Parameters
        ----------
        thr     : Optional threshold value for classification. If provided, additional metrics will be included.
        plot    : wether to show the plot -- defaults False
        save    : wether to save the plot -- defaults False

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
        summary_dict = {"PR-AUC": self.PR_auc(), "ROC-AUC": self.ROC_auc()}

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
                
            print("#------------------------------------------#\n")

        #-------------------------
        if save:

            tag       = f"_at_{thr:.5f}" if thr is not None else ""
            save_path = f"./models/{self.name}_metrics{tag}.csv"

            df_metrics = pd.DataFrame(summary_dict, index=[0])
            df_metrics.to_csv(save_path, index=False)

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
def model_report(metrics : "Metrics", 
                 threshold : float, 
                 train_losses : list = None, 
                 save : bool = False) -> None:

    """
    Performance report of a model

    Parameters
    ----------
    metrics         : Metrics object containing the model's performance metrics.
    threshold       : Threshold value for classification.
    train_losses    : Optional list of training losses for plotting the learning curve -- defaults None
    save            : Whether to save the plot -- defaults False

    Returns
    ----------
    None
    """

    #---------------------------------------------
    sns.set_theme(style="whitegrid")

    if train_losses is not None :
        n_subplots = 3 
    else : 
        n_subplots = 2

    (fig, axes) = plt.subplots(1, n_subplots, figsize=(6 * n_subplots, 5))
    axes        = np.atleast_1d(axes)

    #------------------------
    subplot_idx = 0

    if train_losses is not None:
        learning_curve(axes[subplot_idx], train_losses)           
        subplot_idx += 1

    #--------------
    metrics.distribution_error(threshold, ax=axes[subplot_idx])
    subplot_idx += 1
    CM = metrics.confusion(threshold, plot=True, ax=axes[subplot_idx])

    plt.tight_layout()
    plt.show()

    #--------------
    if save: 
        save_path = f"./models/{metrics.name}_report.png"
        fig.savefig(save_path, bbox_inches='tight')

#================================================================================#
def learning_curve(ax : plt.Axes, train_losses: list) -> None:

    """
    Plot the learning curve for the training losses.

    Parameters
    ----------
    axes            : matplotlib axes object to plot on
    train_losses    : list of training losses for each epoch
    """

    #---------------------------------------------
    ax.plot(train_losses, label='Train Loss (MSE)', color='blue', linewidth=2)
    ax.set_title('Learning Curve', fontsize=14)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Reconstruction Error (MSE)')
    ax.legend()