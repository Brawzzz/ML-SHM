"""
data package : loading and preparation of guided waves datasets.

Modules
-------
GWDataset       : GWDataset class definition

Split           : Split class definition 

preprocessing   : definition of preprocessing functions

UTAH            : UTAH long-term SHM dataset       (pickle)   -> UTAH_data
OGW             : Open Guided Waves temperature ds (HDF5)     -> OGW_load_cycles, OGW_concat, OGW_plot

The functions of each module are re-exported here as a facade allowing 
the rest of the project uses them through `import data`
"""

from .GWDataset import GWDataset
from .Split import Splits, splits
from .preprocessing import prepare, _scaler
from .Utah import UTAH_load, UTAH_files, UTAH_rename_files
from .Ogw  import OGW_load, OGW_concat, OGW_plot


#============================================================================================================================#
#-------------------------------------------------------- FUNCTIONS ---------------------------------------------------------#
#============================================================================================================================#
DATASETS = {
    "UTAH" : UTAH_load,
    "OGW"  : OGW_load,
}

#================================================================================#
def load_dataset(name: str, dir_path: str, **params) -> GWDataset:

    """
    Load a raw dataset by its name.

    Parameters
    ----------
    name     : dataset name -- key of DATASETS
    dir_path : root datasets directory
    params   : loader specific parameters (e.g. nb_sample, path_index for UTAH)
    """

    #---------------------------------------------
    if name not in DATASETS:
        raise ValueError(f"unknown dataset : {name} (available : {list(DATASETS)})")

    return(DATASETS[name](dir_path=dir_path, **params))

#---------------------------------------------
__all__ = [

    "GWDataset",

    "Splits",
    "splits",

    "prepare",
    "_scaler",

    "UTAH_load",
    "UTAH_files",
    "UTAH_rename_files",
    
    "OGW_load",
    "OGW_concat",
    "OGW_plot"

]
