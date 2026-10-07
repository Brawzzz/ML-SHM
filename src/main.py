#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import os
import argparse as ap

import numpy as np

import metrics
import tools

from config import load_experiment
from experiment import run_experiment


#============================================================================================================================#
#--------------------------------------------------------- FUNCTION ---------------------------------------------------------#
#============================================================================================================================#
def arg_parse() -> ap.Namespace:

    """
    Command line :

        python src/main.py run   config/experiments/cae_utah.json [other configs ...]
        python src/main.py tune  config/experiments/cae_utah.json
        python src/main.py train config/experiments/cae_utah.json
        python src/main.py plot  results/CAE_UTAH_pr_auc [--seed 0] [--no-show]
    """

    #---------------------------------------------
    parser = ap.ArgumentParser(description="SHM methods for damage detection : model comparison across datasets")
    sub    = parser.add_subparsers(dest="command", required=True)

    #------------------------------
    p = sub.add_parser("run", help="tuning (if enabled) + final trainings + test evaluation")
    p.add_argument("configs", nargs="+", help="experiment JSON file(s), run one after the other")

    p = sub.add_parser("tune", help="hyperparameter search only (writes best_params.json)")
    p.add_argument("configs", nargs="+", help="experiment JSON file(s)")

    p = sub.add_parser("train", help="final trainings + test evaluation with best_params.json (or defaults)")
    p.add_argument("configs", nargs="+", help="experiment JSON file(s)")

    #------------------------------
    p = sub.add_parser("plot", help="report figure of a trained experiment")
    p.add_argument("experiment_dir", help="results/<experiment name>")
    p.add_argument("--seed", type=int, default=0, help="index of the seed run to plot (default 0)")
    p.add_argument("--no-show", action="store_true", help="only save the figure")

    return parser.parse_args()

#================================================================================#
def plot_experiment(experiment_dir: str, seed: int = 0, show: bool = True) -> None:

    """
    Test report (learning curves, score distribution, confusion matrix) of one seed run
    """

    #---------------------------------------------
    run_dir = os.path.join(experiment_dir, f"seed_{seed}")

    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run not found : {run_dir}")

    scores  = np.load(os.path.join(run_dir, "scores.npz"))
    history = tools.load_json(os.path.join(run_dir, "history.json"))
    name    = f"{os.path.basename(os.path.normpath(experiment_dir))}_seed{seed}"

    #------------------------------
    m = metrics.Metrics(y_true=scores["test_y"], y_pred=scores["test_scores"], name=name)
    m.summary(thr=float(scores["threshold"]), show=True)

    metrics.model_report(m, float(scores["threshold"]),
                         train_losses = history["train_loss"],
                         val_losses   = history["val_loss"],
                         save         = True,
                         save_dir     = experiment_dir,
                         show         = show)

#============================================================================================================================#
#---------------------------------------------------------- MAIN ------------------------------------------------------------#
#============================================================================================================================#
if __name__ == '__main__':

    args = arg_parse()

    #---------------------------------------------
    if args.command in ("run", "tune", "train"):

        for config_path in args.configs:

            cfg = load_experiment(config_path)

            run_experiment(cfg,
                           do_tune  = args.command in ("run", "tune"),
                           do_train = args.command in ("run", "train"))

    #---------------------------------------------
    elif args.command == "plot":

        plot_experiment(args.experiment_dir, seed=args.seed, show=not args.no_show)
