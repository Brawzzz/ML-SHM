#============================================================================================================================#
#--------------------------------------------------------- PACKAGE ----------------------------------------------------------#
#============================================================================================================================#
"""
data package : loading and splitting of the SHM datasets.

Modules
-------
base   : SHMDataset (raw data container) + generic file helpers (explore_file, display_data)
UTAH   : UTAH long-term SHM dataset       (pickle)  -> UTAH_load
OGW    : Open Guided Waves temperature ds (HDF5)    -> OGW_load, OGW_concat, OGW_plot
splits : train / val / test split done ONCE per dataset, scaler fitted on train only

Every loader returns a RAW SHMDataset (no normalisation, no split).
Use `load_dataset(name, dir_path, **params)` to load a dataset by its name.
"""

#---------------------------------------------
from .base   import SHMDataset, explore_file, display_data
from .UTAH   import UTAH_load, UTAH_files, UTAH_rename_files
from .OGW    import OGW_load, OGW_concat, OGW_plot
from .splits import Splits, make_splits, load_or_make_splits, prepare


#============================================================================================================================#
#--------------------------------------------------------- REGISTRY ---------------------------------------------------------#
#============================================================================================================================#
DATASETS = {
    "UTAH" : UTAH_load,
    "OGW"  : OGW_load,
}

#================================================================================#
def load_dataset(name: str, dir_path: str, **params) -> SHMDataset:

    """
    Load a raw dataset by its name.

    Parameters
    ----------
    name     : dataset name, key of DATASETS
    dir_path : root datasets directory
    params   : loader specific parameters (e.g. nb_sample, path_index for UTAH)
    """

    #---------------------------------------------
    if name not in DATASETS:
        raise ValueError(f"unknown dataset : {name} (available : {list(DATASETS)})")

    return DATASETS[name](dir_path=dir_path, **params)

#---------------------------------------------
__all__ = [
    "SHMDataset",
    "Splits",
    "DATASETS",
    "load_dataset",
    "make_splits",
    "load_or_make_splits",
    "prepare",
    "explore_file",
    "display_data",
    "UTAH_load",
    "UTAH_files",
    "UTAH_rename_files",
    "OGW_load",
    "OGW_concat",
    "OGW_plot",
]
