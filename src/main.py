#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import argparse as ap

import config

from run import run, plot_run

    
#============================================================================================================================#
#-------------------------------------------------------- FUNCTION ----------------------------------------------------------#
#============================================================================================================================#
def arg_parse() -> ap.Namespace:

    """
    Read the command line : one command and its arguments.

    Commands
    ----------
        run   <configs ...>         tuning (if tuning.enabled) + final trainings + test evaluation
        tune  <configs ...>         hyperparameter search only -> writes best_params.json
        train <configs ...>         final trainings + test evaluation with best_params.json
                                    (or the model defaults if the run was never tuned)

        plot  <run_dir> [--seed k] [--no-show]
                                    report figure of a trained run (see run.plot_run)

    Several config files can be given to run / tune / train : they are executed one after the other.

    Returns
    ----------
    args : (argparse.Namespace) parsed arguments

           -> args.command  : "run" | "tune" | "train" | "plot"
           -> args.configs  : (list[str]) config paths -- run / tune / train
           -> args.run_dir, args.seed, args.

    Examples
    ----------
        python src/main.py run   ./config/runs/cae_utah.json config/runs/cae_ogw.json
        python src/main.py tune  ./config/runs/cae_utah.json
        python src/main.py train ./config/runs/cae_utah.json
        python src/main.py plot  ./results/CAE_UTAH_pr_auc --seed 1 --no-show
    """

    #---------------------------------------------
    parser = ap.ArgumentParser(description="SHM damage detection : model comparison across datasets")
    sub    = parser.add_subparsers(dest="command", required=True)

    for (name, help_txt) in (("run",   "tuning (if enabled) + final trainings + test evaluation"),
                             ("tune",  "hyperparameter search only"),
                             ("train", "final trainings with best_params.json (or defaults)")):
        
        p = sub.add_parser(name, help=help_txt)
        p.add_argument("configs", nargs="+", help="run JSON file(s)")

    p = sub.add_parser("plot", help="report figure of a trained run")

    p.add_argument("run_dir", help="results/<run name>")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-show", action="store_true")

    return parser.parse_args()

#============================================================================================================================#
#---------------------------------------------------------- MAIN ------------------------------------------------------------#
#============================================================================================================================#
if __name__ == "__main__":

    args = arg_parse()

    #--------------------
    if args.command == "plot":

        plot_run(args.run_dir, seed=args.seed, show=not args.no_show)

    #--------------------
    else:

        for config_path in args.configs:

            run_config = config.load_run(config_path)

            summary = run(config   = run_config,
                          do_tune  = args.command in ("run", "tune"),
                          do_train = args.command in ("run", "train"))