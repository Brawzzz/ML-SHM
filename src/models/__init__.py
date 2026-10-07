#============================================================================================================================#
#--------------------------------------------------------- PACKAGE ----------------------------------------------------------#
#============================================================================================================================#
"""
models package : anomaly detectors sharing the BaseDetector interface.

To add a model :
  1. write a BaseDetector subclass (DEFAULTS, SEARCH_SPACE, forward, training_step, anomaly_score)
  2. register it in MODELS below
Nothing else changes : training, tuning and evaluation are generic.
"""

#---------------------------------------------
from .base import BaseDetector
from .CAE  import ConvAutoEncoder, CAE_plot


#============================================================================================================================#
#--------------------------------------------------------- REGISTRY ---------------------------------------------------------#
#============================================================================================================================#
MODELS = {
    "CAE" : ConvAutoEncoder,
}

#================================================================================#
def get_model(name: str) -> type[BaseDetector]:

    """Return the model class registered under `name`"""

    #---------------------------------------------
    if name not in MODELS:
        raise ValueError(f"unknown model : {name} (available : {list(MODELS)})")

    return MODELS[name]

#---------------------------------------------
__all__ = ["BaseDetector", "ConvAutoEncoder", "CAE_plot", "MODELS", "get_model"]
