#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import torch
import torch.nn as nn


#============================================================================================================================#
#---------------------------------------------------------- CLASS -----------------------------------------------------------#
#============================================================================================================================#
class BaseDetector(nn.Module):

    """
    Common interface of every PyTorch anomaly detector.

    A model OWNS its hyperparameters :
      - DEFAULTS     : value of every hyperparameter (architecture + optimisation)
      - SEARCH_SPACE : how each tunable hyperparameter is sampled by Optuna

    The tuned values are NOT stored in the model or in the config : they are an
    OUTPUT of the tuning, saved per (model, dataset) in results/<experiment>/best_params.json.

    A subclass implements :
      - __init__(input_length, **architecture_params)
      - forward(x)
      - training_step(batch)  -> scalar loss minimised by the generic training loop
      - anomaly_score(x)      -> one score per signal, the higher the more anomalous

    SEARCH_SPACE format (JSON friendly, can be overridden from the experiment config) :
      {"lr"         : {"type": "float", "low": 1e-4, "high": 1e-2, "log": true},
       "n_layers"   : {"type": "int",   "low": 2,    "high": 4},
       "batch_size" : {"type": "categorical", "choices": [16, 32, 64]}}
    """

    NAME          : str  = "base"
    DEFAULTS      : dict = {}
    SEARCH_SPACE  : dict = {}

    # hyperparameters used by the training loop, the others are given to __init__
    TRAINING_KEYS = ("lr", "batch_size", "optimizer", "weight_decay")

    #================================================================================#
    def __init__(self, input_length: int):

        super().__init__()
        self.input_length = input_length

    #================================================================================#
    def training_step(self, batch: list) -> torch.Tensor:
        raise NotImplementedError

    #================================================================================#
    def validation_loss(self, batch: list) -> torch.Tensor:

        """Loss on a validation batch (called under torch.no_grad), same as training by default"""

        return self.training_step(batch)

    #================================================================================#
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    #================================================================================#
    @classmethod
    def resolve_params(cls, params: dict = None) -> dict:

        """
        DEFAULTS updated with `params`. Unknown names raise an error (typo protection).
        """

        #---------------------------------------------
        params  = dict(params or {})
        unknown = set(params) - set(cls.DEFAULTS)

        if unknown:
            raise ValueError(f"[{cls.NAME}] unknown hyperparameter(s) : {sorted(unknown)} "
                             f"(available : {sorted(cls.DEFAULTS)})")

        return {**cls.DEFAULTS, **params}

    #================================================================================#
    @classmethod
    def split_params(cls, params: dict) -> tuple[dict, dict]:

        """Split a full parameter dict into (architecture params, training params)"""

        arch  = {k: v for k, v in params.items() if k not in cls.TRAINING_KEYS}
        train = {k: v for k, v in params.items() if k in cls.TRAINING_KEYS}

        return (arch, train)

    #================================================================================#
    @classmethod
    def build(cls, input_length: int, params: dict = None) -> "BaseDetector":

        """Instantiate the model from a (partial) parameter dict"""

        (arch, _) = cls.split_params(cls.resolve_params(params))

        return cls(input_length=input_length, **arch)

    #================================================================================#
    @classmethod
    def suggest_params(cls, trial, fixed: dict = None, search_space: dict = None) -> dict:

        """
        Sample a full parameter dict for an Optuna trial.

        Parameters
        ----------
        trial        : optuna.Trial
        fixed        : hyperparameters fixed by the experiment config (not tuned)
        search_space : overrides of SEARCH_SPACE from the experiment config
        """

        #---------------------------------------------
        fixed = dict(fixed or {})
        space = {**cls.SEARCH_SPACE, **(search_space or {})}

        unknown = set(space) - set(cls.DEFAULTS)
        if unknown:
            raise ValueError(f"[{cls.NAME}] search space on unknown hyperparameter(s) : {sorted(unknown)}")

        #---------------------------------------------
        sampled = {}

        for (name, spec) in space.items():

            if name in fixed:
                continue

            kind = spec["type"]

            if kind == "float":
                sampled[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))

            elif kind == "int":
                sampled[name] = trial.suggest_int(name, spec["low"], spec["high"],
                                                  step=spec.get("step", 1), log=spec.get("log", False))

            elif kind == "categorical":
                sampled[name] = trial.suggest_categorical(name, spec["choices"])

            else:
                raise ValueError(f"[{cls.NAME}] unknown search space type '{kind}' for '{name}'")

        return cls.resolve_params({**sampled, **fixed})
