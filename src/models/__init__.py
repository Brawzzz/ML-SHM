#============================================================================================================================#
#--------------------------------------------------------- PACKAGE ----------------------------------------------------------#
#============================================================================================================================#
"""
models package : definition of differents models.

Modules
-------
BaseDetector : Common class for all anomaly detector models

CAE : Convolutional AutoEncoder -- used for detection

To add a model :
  1. write a BaseDetector subclass (DEFAULTS, SEARCH_SPACE, forward, training_step, anomaly_score)
  2. add it to MODELS 

Nothing else changes : training, tuning and evaluation are generic.

The functions below are re-exported here as a facade, 
allowning the rest of the project uses them through `import models`
"""

#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
from .BaseDetector import BaseDetector

from .CAE import ConvAutoEncoder
from .AE import AutoEncoder


#---------------------------------------------
MODELS = {
    "CAE" : ConvAutoEncoder,
    "AE"  : AutoEncoder,
}

__all__ = [
    
    "Base Detector",

    "ConvAutoEncoder",
    
]

#================================================================================#
def get_model(name: str) -> type[BaseDetector]:

    """
    Return the model class registered under `name`
    """

    #---------------------------------------------
    if name not in MODELS:
        raise ValueError(f"unknown model : {name} (available : {list(MODELS)})")

    return MODELS[name]


