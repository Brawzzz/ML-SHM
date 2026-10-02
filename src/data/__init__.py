#============================================================================================================================#
#--------------------------------------------------------- PACKAGE ----------------------------------------------------------#
#============================================================================================================================#
"""
data package : loading and preparation of the SHM datasets.

Modules
-------
base : generic file helpers (explore_file, display_data)
utah : UTAH long-term SHM dataset       (pickle)   -> UTAH_data
ogw  : Open Guided Waves temperature ds (HDF5)     -> OGW_load_cycles, OGW_concat, OGW_plot

The functions below are re-exported here as a facade, so the rest of the
project keeps using them through `import data` exactly as before
(e.g. data.UTAH_data(...), data.explore_file(...)), while each dataset lives
in its own module.
"""

#---------------------------------------------
from .base import explore_file, display_data
from .UTAH import UTAH_load
from .OGW  import OGW_load, OGW_concat, OGW_plot

#---------------------------------------------
__all__ = [
    "explore_file",
    "display_data",
    "UTAH_load",
    "OGW_load",
    "OGW_concat",
    "OGW_plot",
]
