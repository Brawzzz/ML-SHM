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
import copy
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as skm

from matplotlib.patches import FancyBboxPatch

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

#================================================================================#
_SECTIONS = {"encoder"     : "Encoder",
             "to_latent"   : "Code",
             "from_latent" : "Code",
             "decoder"     : "Decoder"}

_COLORS   = {"Input"   : "#9aa5b1",
             "Encoder" : "#4c78a8",
             "Code"    : "#e45756",
             "Decoder" : "#54a24b",
             "Output"  : "#9aa5b1"}

_MERGED   = (nn.ReLU, nn.LeakyReLU, nn.GELU, nn.ELU, nn.Sigmoid, nn.Tanh, nn.Dropout)

#================================================================================#
def _layer_label(module: nn.Module) -> str:

    """
    Short label of a layer for the architecture diagram
    """

    #---------------------------------------------
    if isinstance(module, nn.ConvTranspose1d):
        return f"ConvT1d {module.in_channels}→{module.out_channels}\nk={module.kernel_size[0]}, s={module.stride[0]}"

    if isinstance(module, nn.Conv1d):
        stride = f", s={module.stride[0]}" if module.stride[0] > 1 else ""
        return f"Conv1d {module.in_channels}→{module.out_channels}\nk={module.kernel_size[0]}{stride}"

    if isinstance(module, nn.MaxPool1d):
        return f"MaxPool\n÷{module.kernel_size}"

    if isinstance(module, nn.Upsample):
        return f"Upsample\n×{int(module.scale_factor)}"

    if isinstance(module, nn.Linear):
        return f"Dense\n{module.in_features}→{module.out_features}"

    return type(module).__name__

#================================================================================#
def _fmt_params(n: int) -> str:

    if n >= 1e6: return f"{n / 1e6:.2f}M"
    if n >= 1e3: return f"{n / 1e3:.1f}k"
    return str(n)

#================================================================================#
def _trace_layers(model: nn.Module) -> tuple[list[dict], tuple]:

    """
    Run one dummy signal through `model` and record every leaf layer :
    section, label, output shape, number of parameters.
    Activations / dropout are merged into the previous layer.

    Returns
    ----------
    (blocks, out_shape) : (list[dict], tuple) one dict per drawn block | shape of the model output
    """

    #---------------------------------------------
    net     = copy.deepcopy(model).cpu().eval()
    records = []
    hooks   = []

    for (name, module) in net.named_modules():

        if list(module.children()) or isinstance(module, nn.Identity):
            continue

        hook = module.register_forward_hook(
            lambda mod, inp, out, name=name: records.append((name, mod, tuple(out.shape[1:]))))
        hooks.append(hook)

    #--------------------
    x = torch.zeros(1, getattr(net, "in_channels", 1), net.input_length)

    with torch.no_grad():
        out_shape = tuple(net(x).shape[1:])

    for hook in hooks:
        hook.remove()

    #---------------------------------------------
    blocks = []

    for (name, module, shape) in records:

        section = _SECTIONS.get(name.split(".")[0], name.split(".")[0].capitalize())
        params  = sum(p.numel() for p in module.parameters(recurse=False))
        frozen  = params > 0 and not any(p.requires_grad for p in module.parameters(recurse=False))

        if isinstance(module, _MERGED) and blocks and blocks[-1]["section"] == section:
            blocks[-1]["act"] = type(module).__name__
            continue

        blocks.append({"section" : section,
                       "label"   : _layer_label(module),
                       "shape"   : shape,
                       "params"  : params,
                       "frozen"  : frozen,
                       "act"     : None})

    return (blocks, out_shape)

#================================================================================#
def model_architecture(model         : nn.Module,
                       save_dir      : str  = None,
                       name          : str  = None,
                       print_summary : bool = True,
                       show          : bool = True) -> plt.Figure:

    """
    Block diagram of a BaseDetector (e.g. a trained model from run.load_trained) :

        - one block per layer, height ~ log(signal length), width ~ log(channels)
        - colored by section : Encoder | Code | Decoder
        - under each block : layer type, output shape, number of parameters
        - the latent code (smallest representation) is outlined in black
        - frozen layers (requires_grad = False, e.g. transfer learning) are hatched

    The model is traced with one zero signal of shape (1, in_channels, input_length),
    on a copy on CPU : the given model is left untouched.

    Args
    ----------
    model         : (BaseDetector) model to draw -- needs `input_length` (and `in_channels`)
    save_dir      : (str) directory where the figure is saved as <name>_architecture.png -- None = no save
    name          : (str) name used in the title and file name -- defaults model.NAME
    print_summary : (bool) also print a layer-by-layer table in the console -- defaults True
    show          : (bool) display the figure -- defaults True

    Returns
    ----------
    fig : (plt.Figure) matplotlib figure

    Example
    ----------
        (model, _) = run.load_trained("./results/CAE_OGW_sawant_noTL/seed_0")
        plots.model_architecture(model, save_dir="./results/CAE_OGW_sawant_noTL")
    """

    #---------------------------------------------
    name               = name or getattr(model, "NAME", type(model).__name__)
    (blocks, out_shape) = _trace_layers(model)

    in_shape = (getattr(model, "in_channels", 1), model.input_length)

    blocks = ([{"section": "Input",  "label": "Input",  "shape": in_shape,  "params": 0, "act": None}] + blocks +
              [{"section": "Output", "label": "Output", "shape": out_shape, "params": 0, "act": None}])

    #--------------------------------------------- code = smallest representation
    sizes    = [int(np.prod(b["shape"])) for b in blocks[1:-1]]
    code_idx = 1 + int(np.argmin(sizes))

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    #--------------------------------------------- console summary
    if print_summary:

        print(f"\n#{f' {name} architecture '.center(76, '-')}#")
        print(f"  {'section':<9} {'layer':<28} {'output shape':<16} {'params':>10}")

        for (i, b) in enumerate(blocks):
            layer = b["label"].replace("\n", " ") + (f" + {b['act']}" if b["act"] else "")
            mark  = "  <- code" if i == code_idx else ""
            print(f"  {b['section']:<9} {layer:<28} {' x '.join(map(str, b['shape'])):<16} {b['params']:>10,}{mark}")

        print(f"\n  total : {total:,} parameters | trainable : {trainable:,} | frozen : {total - trainable:,}")
        print("#" + "-" * 76 + "#\n")

    #--------------------------------------------- block geometry
    def dims(shape):
        (ch, ln) = shape if len(shape) == 2 else (1, shape[0])
        return (ch, ln)

    max_len = max(dims(b["shape"])[1] for b in blocks)
    max_ch  = max(dims(b["shape"])[0] for b in blocks)

    def height(ln): return 0.5 + 2.5 * np.log1p(ln) / np.log1p(max_len)
    def width(ch):  return 0.18 + 0.55 * np.log2(ch + 1) / np.log2(max_ch + 1)

    #---------------------------------------------
    sns.set_theme(style="white")

    gap         = 0.45
    total_width = sum(width(dims(b["shape"])[0]) + gap for b in blocks)
    (fig, ax)   = plt.subplots(figsize=(max(10, 1.15 * total_width), 6.5))

    x        = 0.0
    centers  = []
    sections = {}

    for (i, b) in enumerate(blocks):

        (ch, ln) = dims(b["shape"])
        (w, h)   = (width(ch), height(ln))
        color    = _COLORS.get(b["section"], "#bab0ac")

        ax.add_patch(FancyBboxPatch((x, -h / 2), w, h,
                                    boxstyle  = "round,pad=0.02,rounding_size=0.05",
                                    facecolor = color, alpha=0.85,
                                    edgecolor = "black" if i == code_idx else "white",
                                    linewidth = 2.5 if i == code_idx else 1.0,
                                    hatch     = "///" if b.get("frozen") else None))

        cx = x + w / 2
        centers.append(cx)

        #----------- shape above, layer + params below
        ax.text(cx, h / 2 + 0.08, " × ".join(map(str, b["shape"])),
                ha="center", va="bottom", fontsize=7, rotation=90)

        label = b["label"] + (f"\n+ {b['act']}" if b["act"] else "")
        if b["params"]:
            label += f"\n{_fmt_params(b['params'])} p." + (" (frozen)" if b.get("frozen") else "")

        ax.text(cx, -1.65, label, ha="center", va="top", fontsize=7, linespacing=1.15)

        sections.setdefault(b["section"], []).append((x, x + w, b["params"]))
        x += w + gap

    #--------------------------------------------- arrows between blocks
    for (c0, c1) in zip(centers[:-1], centers[1:]):
        ax.annotate("", xy=(c1, 0), xytext=(c0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0.6, alpha=0.5), zorder=0)

    #--------------------------------------------- section brackets
    for (section, spans) in sections.items():

        if section in ("Input", "Output"):
            continue

        (x0, x1) = (spans[0][0], spans[-1][1])
        n_params = sum(s[2] for s in spans)
        color    = _COLORS.get(section, "#bab0ac")

        ax.plot([x0, x1], [2.35, 2.35], color=color, lw=3, solid_capstyle="round")
        ax.text((x0 + x1) / 2, 2.45, f"{section} — {_fmt_params(n_params)} params",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color=color)

    #---------------------------------------------
    ax.set_xlim(-gap, x)
    ax.set_ylim(-3.0, 3.0)
    ax.axis("off")

    frozen = f" | frozen : {_fmt_params(total - trainable)}" if total != trainable else ""
    ax.set_title(f"{name} — {total:,} parameters{frozen}  |  input {' × '.join(map(str, in_shape))}"
                 f"  →  code {' × '.join(map(str, blocks[code_idx]['shape']))}",
                 fontsize=12, fontweight="bold")

    fig.tight_layout()

    #-------------------------
    if save_dir is not None:

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{name}_architecture.png")

        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Architecture saved at : {save_path}")

    #-------------------------
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig