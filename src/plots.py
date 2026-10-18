"""
This module provides functions that plots multiple kind of graphs
to represent the performances of a model according to its computed metrics

    - distribution_error()
    - PR_curve()
    - ROC_curve()
    - confusion_matrix()
    - learning_curve()

    - model_report() : combines all the graphs in a single Figure
""" 

#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as skm

from Metrics import Metrics


#============================================================================================================================#
#-------------------------------------------------------- FUNCTIONS ---------------------------------------------------------#
#============================================================================================================================#
def _get_ax(ax: plt.Axes, figsize: tuple) -> tuple[plt.Axes, bool]:
 
    """
    Return the axes to draw on : `ax` itself if given, otherwise a new figure and axes.

    Boolean `created` indicates to the function caller whether the figure was created here : 
    only in this case should the caller display it (plt.show)
    A figure passed by another caller (a subgraph from `model_report`) is displayed by its owner.

    Args
    ----------
    ax      : (plt.Axes) axes to draw on -- None to create a new figure
    figsize : (tuple) (width, height) of the new figure in inches -- ignored if `ax` is given

    Returns
    ----------
    (ax, created) : (tuple) tuple containing
                    ax      -> (plt.Axes) the given axes, or the newly created one
                    created -> (bool) True if a new figure was created by this function
    """

    #---------------------------------------------
    if ax is None:
        
        sns.set_theme(style="whitegrid")
        (_, ax) = plt.subplots(figsize=figsize)

        return (ax, True)
 
    return (ax, False)

#================================================================================#
def distribution_error(metrics   : Metrics, 
                       thr       : float, 
                       ax        : plt.Axes = None,
                       show      : bool = True) -> plt.Axes:

    """
    MSE histogramme healty / dammaged + threshold line

    Args
    ----------
    metrics : (Metrics) Metrics object containing the model's performance metrics
    thr     : (float) Threshold value for classification.
    ax      : (plt.Axes) Axes object for plotting.
    show    : (bool) whether to show the plot

    Returns
    ----------
    ax : (plt.Axes) ploted axes 
    """
    
    #---------------------------------------------
    (ax, created) = _get_ax(ax, figsize=(7, 5))

    scores = metrics.y_score[metrics.y_score > 0]
    bins   = np.logspace(np.log10(scores.min()), np.log10(scores.max()), 60)

    healthy = metrics.y_score[metrics.y_true == 0]
    damaged = metrics.y_score[metrics.y_true == 1]

    sns.histplot(healthy, bins=bins, color="green", alpha=0.6, label="Healthy", ax=ax, stat="probability")
    sns.histplot(damaged, bins=bins, color="red",   alpha=0.5, label="Damaged", ax=ax, stat="probability")
    ax.set_xscale("log")

    ax.axvline(thr, color="black", linestyle="dashed", linewidth=2, label="Warning threshold")
    ax.set_title("Reconstruction error separation", fontsize=14)
    ax.set_xlabel("Anomaly score"); ax.set_ylabel("Density")
    ax.legend()

    if created and show:
        plt.show()  

    return ax

#================================================================================#
def PR_curve(metrics : Metrics, 
             ax      : plt.Axes = None, 
             show    : bool     = True) -> plt.Axes:

    """
    plot the PR-curve (Precision-Recall) from `metrics`

    Args
    ----------
    metrics : (Metrics) Metrics object containing the model's performance metrics
    ax      : (plt.Axes) Axes to plot on -- if None, a new figure and axes will be created -- default None.
    show    : (bool) Whether to display the plot -- default True.

    Returns
    ----------
    ax : (plt.Axes) ploted axes
    """

    #---------------------------------------------
    (ax, created) = _get_ax(ax, figsize=(7, 6))

    base                   = metrics.y_true.mean()
    (precision, recall, _) = skm.precision_recall_curve(metrics.y_true, metrics.y_score)

    #-------------------------
    label = f"Baseline ({base:.3f})" if created else None

    ax.plot(recall, precision, lw=2, label=f"{metrics.name or 'PR'} (AP={metrics.PR_auc():.3f})")
    ax.axhline(base, color="grey", lw=1, ls="--",label=label)

    ax.set_title("Precision-Recall")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    ax.legend(loc="lower left")

    #-------------------------
    if created and show:
        plt.show()

    return ax

#================================================================================#
def ROC_curve(metrics : Metrics, 
              ax      : plt.Axes = None, 
              show    : bool     = True) -> plt.Axes:

    """
    plot the ROC-curve  from `metrics`

    Args
    ----------
    metrics : (Metrics) Metrics object containing the model's performance metrics
    ax      : (plt.Axes) Axes to plot on -- if None, a new figure and axes will be created -- default None.
    show    : (bool) Whether to display the plot -- default True.

    Returns
    ----------
    ax : (plt.Axes) ploted axes
    """

    #---------------------------------------------
    (ax, created) = _get_ax(ax, figsize=(7, 6))

    roc_auc         = metrics.ROC_auc()
    (f_pr, t_pr, _) = skm.roc_curve(metrics.y_true, metrics.y_score)

    label = "Random" if created else None

    #-------------------------
    ax.plot(f_pr, t_pr, lw=2, label=f"{metrics.name or 'ROC'} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--", label=label)

    ax.set_title("ROC")
    ax.set_xlabel("FPR (False positive rate)")
    ax.set_ylabel("TPR (True positive rate -- Recall)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    ax.legend(loc="lower right")

    #-------------------------
    if created and show:
        plt.show()
 
    return ax

#================================================================================#
def confusion_matrix(metrics : Metrics, 
                     thr     : float, 
                     ax      : plt.Axes = None, 
                     show    : bool = True) -> plt.Axes:
 
    """
    Plot the confusion matrix heatmap for a given threshold
 
    Args
    ----------
    metrics : (Metrics) Metrics object containing the model's performance metrics
    thr     : (float) alert threshold
    ax      : (plt.Axes) axes to plot on -- if None, a new figure is created
    show    : (bool) display the figure if it was created here -- defaults True

    Returns
    ----------
    ax : (plt.Axes) ploted axes
    """
 
    #---------------------------------------------
    (ax, created) = _get_ax(ax, figsize=(6, 5))
 
    sns.heatmap(data        = metrics.confusion(thr),
                annot       = True,
                fmt         = "d",
                cmap        = "Blues",
                ax          = ax,
                xticklabels = ["Predicted Healthy", "Predicted Damaged"],
                yticklabels = ["Healthy", "Damaged"])
 
    ax.set_title("Confusion Matrix", fontsize=14)

    #-------------------------
    if created and show:
        plt.show()

    return ax

#================================================================================#
def learning_curve(ax           : plt.Axes, 
                   train_losses : list, 
                   val_losses   : list = None) -> plt.Axes:
 
    """
    Training (and validation) loss for each epoch
 
    Args
    ----------
    ax           : (plt.Axes) axes to plot on
    train_losses : (list[float]) training loss of each epoch
    val_losses   : (list[float]) validation loss of each epoch -- defaults None

    Returns
    ----------
    ax : (plt.Axes) ploted axes
    """
 
    #---------------------------------------------
    ax.plot(train_losses, label="Train loss", color="blue", linewidth=2)
 
    if val_losses is not None:
        ax.plot(val_losses, label="Validation loss", color="orange", linewidth=2)
 
    ax.set_title("Learning curve", fontsize=14)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend()
 
    return ax

#================================================================================#
def model_report(metrics      : Metrics,
                 thr          : float,
                 train_losses : list = None,
                 val_losses   : list = None,
                 save_dir     : str  = None,
                 show         : bool = True) -> plt.Figure:
 
    """
    Performance report : [learning curve] + score distribution + confusion matrix
 
    Args
    ----------
    metrics      : (Metrics) Metrics object containing the model's performance metrics
    thr          : (float) alert threshold
    train_losses : (list[float]) training loss of each epoch -- defaults None (no learning curve)
    val_losses   : (list[float]) validation loss of each epoch -- defaults None
    save_dir     : (str) output directory -- defaults None
    show         : (bool) whether to display the figure -- defaults True
 
    Returns
    ----------
    fig : (plt.Figure) matplotlib figure
    """
 
    #---------------------------------------------
    sns.set_theme(style="whitegrid")
 
    n_subplots  = 3 if train_losses is not None else 2
    (fig, axes) = plt.subplots(1, n_subplots, figsize=(6 * n_subplots, 5))
    axes        = list(np.atleast_1d(axes))
 
    #-------------------------
    if train_losses is not None:
        learning_curve(axes.pop(0), train_losses, val_losses)
 
    distribution_error(metrics, thr, ax=axes[0])
    confusion_matrix(metrics, thr, ax=axes[1])
 
    fig.tight_layout()
 
    #-------------------------
    if save_dir is not None:

        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{metrics.name}_report.png")

        fig.savefig(save_path, bbox_inches="tight")
        print(f"Report saved at : {save_path}")

    #-------------------------
    if show:
        plt.show()
    else:
        plt.close(fig)
 
    return fig