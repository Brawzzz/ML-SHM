#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import numpy as np
import pandas as pd

from dataclasses import dataclass, field


#============================================================================================================================#
#----------------------------------------------------------- CLASS ----------------------------------------------------------#
#============================================================================================================================#
@dataclass
class GWDataset :

    """
    This class decribe datasets that contains guided-waves measurements

    Attributes:
    ----------
    name   : dataset name (e.g. "OGW")
    
    X      : signals -- shape (n_signals, signal_length) -- float32
    y      : labels -- shape (n_signals,) -- 0 = healthy | 1 = damaged

    groups : acquisition block of each signal, shape (n_signals,) -- int 
             Signals of the same block are strongly correlated (same file / consecutive cycles) 
             and are always kept in the same split. 
             Block ids are ordered chronologically (block 0 = oldest).

    meta   : one row per signal with dataset-specific information (temperature, damage tag ...)
    """

    #---------------------------------------------
    name   : str
    X      : np.ndarray
    y      : np.ndarray
    groups : np.ndarray
    meta   : pd.DataFrame = field(default_factory=pd.DataFrame)


    #================================================================================#
    def __post_init__(self):

        """
        verification of data validity : shape, dimensions 

        execute automaticly after the instanciation of a GWDataset object 
        """

        #---------------------------------------------
        self.X      = np.asarray(self.X, dtype=np.float32)
        self.y      = np.asarray(self.y, dtype=np.int64)
        self.groups = np.asarray(self.groups, dtype=np.int64)

        if self.X.ndim != 2:
            raise ValueError(f"X must be 2D (n_signals, signal_length) -- shape : {self.X.shape}")

        if not (len(self.X) == len(self.y) == len(self.groups)):
            raise ValueError(f"X, y and groups must have the same length : "
                             f"{len(self.X)}, {len(self.y)}, {len(self.groups)}")

        if len(self.meta) and len(self.meta) != len(self.X):
            raise ValueError(f"meta must have one row per signal : {len(self.meta)} != {len(self.X)}")

    #================================================================================#
    def __str__(self):
        return self.describe()

    #================================================================================#
    @property
    def signal_length(self) -> int:
        return self.X.shape[1]

    #================================================================================#
    def describe(self) -> str:

        """
        Return and print a description of the GWDataset object

        Example:
        ----------
        //-------------------- GWDataset | UTAH dataset --------------------//
        Loaded signals : 59283 (healthy : 20022 | damaged : 39261)
        signal length  : 2000
        blocks         : 1189

        Parameters:
        ----------
        None

        Returns:
        ----------
        desc : (str) descrption of the GWDataset instance
        """
        
        #---------------------------------------------
        n_healthy = int((self.y == 0).sum())
        n_damaged = int((self.y == 1).sum())

        lines = [f"\n//-------------------- GWDataset | {self.name} dataset --------------------//\n",
                 f"Loaded signals : {len(self.X)} (healthy : {n_healthy} | damaged : {n_damaged})",
                 f"signal length  : {self.signal_length}",
                 f"blocks         : {len(np.unique(self.groups))}"]

        desc = "\n".join(lines) + "\n"

        return desc